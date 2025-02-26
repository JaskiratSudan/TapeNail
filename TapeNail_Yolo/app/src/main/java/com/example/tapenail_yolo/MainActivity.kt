package com.example.tapenail_yolo

import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import android.graphics.Paint
import android.graphics.Color
import android.graphics.Canvas
import android.util.Log
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

@Suppress("DEPRECATION", "NAME_SHADOWING")
class MainActivity : AppCompatActivity() {

    private lateinit var bitmap: Bitmap
    private lateinit var imageView: ImageView
    private lateinit var cameraDevice: CameraDevice
    private lateinit var handler: Handler
    private lateinit var textureView: TextureView
    private lateinit var cameraManager: CameraManager
    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>

    // Reusable objects
    private val inputBuffer = ByteBuffer.allocateDirect(4 * 256 * 256 * 3).order(ByteOrder.nativeOrder())
    private val outputBuffer = Array(1) { Array(5) { FloatArray(1344) } }
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 5f
        textSize = 40f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        // Initialize imageView
        imageView = findViewById(R.id.imageView)

        // Other initialization code
        get_permission()
        tflite = Interpreter(loadModelFile())
        labels = loadLabelList()

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        textureView = findViewById(R.id.texture)
        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                open_camera(surface)
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                handler.post {
                    val results = runInference(bitmap)
                    drawDetectionResults(results)
                }
            }
        }

        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
    }

    @SuppressLint("MissingPermission")
    private fun open_camera(surface: SurfaceTexture) {
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera

                val surfaceTexture = textureView.surfaceTexture
                val surface = Surface(surfaceTexture)

                val captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(captureRequest.build(), null, null)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Log.e("MainActivity", "Camera session configuration failed")
                    }
                }, handler)
            }

            override fun onDisconnected(camera: CameraDevice) {
                Log.e("MainActivity", "Camera disconnected")
            }

            override fun onError(camera: CameraDevice, error: Int) {
                Log.e("MainActivity", "Camera error: $error")
            }
        }, handler)
    }

    private fun get_permission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    private data class DetectionResult(
        val xmin: Float,
        val ymin: Float,
        val xmax: Float,
        val ymax: Float,
        val classId: Int,
        val confidence: Float
    )

    private fun drawDetectionResults(results: List<DetectionResult>) {
        runOnUiThread {
            val canvas = Canvas(bitmap)
            for (result in results) {
                // Convert normalized coordinates to pixel coordinates
                val rect = RectF(
                    result.xmin * bitmap.width,
                    result.ymin * bitmap.height,
                    result.xmax * bitmap.width,
                    result.ymax * bitmap.height
                )

                // Draw bounding box
                canvas.drawRect(rect, paint)

                // Draw label (always "Pattern" for single class)
                canvas.drawText("Pattern (${result.confidence})", rect.left, rect.top - 10, paint)
            }
            imageView.setImageBitmap(bitmap)
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        inputBuffer.rewind() // Reset buffer position
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        val intValues = IntArray(256 * 256)
        resizedBitmap.getPixels(intValues, 0, 256, 0, 0, 256, 256)
        val floatBuffer = inputBuffer.asFloatBuffer()
        for (value in intValues) {
            floatBuffer.put(((value shr 16) and 0xFF) / 255.0f) // Normalize R
            floatBuffer.put(((value shr 8) and 0xFF) / 255.0f)  // Normalize G
            floatBuffer.put((value and 0xFF) / 255.0f)          // Normalize B
        }
    }

    private fun runInference(bitmap: Bitmap): List<DetectionResult> {
        convertBitmapToByteBuffer(bitmap)
        tflite.run(inputBuffer, outputBuffer)

        val results = mutableListOf<DetectionResult>()
        val output = outputBuffer[0]

        for (i in 0 until 1344) {
            val xCenter = output[0][i]
            val yCenter = output[1][i]
            val width = output[2][i]
            val height = output[3][i]
            val confidence = output[4][i]
            val classId = 0 // Force classId to 0 for the single class

            if (confidence > 0.75) { // Confidence threshold
                results.add(
                    DetectionResult(
                        xmin = xCenter - width / 2,
                        ymin = yCenter - height / 2,
                        xmax = xCenter + width / 2,
                        ymax = yCenter + height / 2,
                        classId = classId,
                        confidence = confidence
                    )
                )
            }
        }

        // Apply NMS
        val boxes = results.map { RectF(it.xmin, it.ymin, it.xmax, it.ymax) }
        val scores = results.map { it.confidence }
        val selectedIndices = applyNMS(boxes, scores, iouThreshold = 0.5f)

        return selectedIndices.map { results[it] }
    }

    private fun applyNMS(boxes: List<RectF>, scores: List<Float>, iouThreshold: Float = 0.5f): List<Int> {
        val selectedIndices = mutableListOf<Int>()
        val sortedIndices = scores.indices.sortedByDescending { scores[it] }

        for (i in sortedIndices) {
            var shouldSelect = true
            for (j in selectedIndices) {
                if (iou(boxes[i], boxes[j]) > iouThreshold) {
                    shouldSelect = false
                    break
                }
            }
            if (shouldSelect) {
                selectedIndices.add(i)
            }
        }

        return selectedIndices
    }

    private fun iou(boxA: RectF, boxB: RectF): Float {
        val xA = max(boxA.left, boxB.left)
        val yA = max(boxA.top, boxB.top)
        val xB = min(boxA.right, boxB.right)
        val yB = min(boxA.bottom, boxB.bottom)

        val interArea = max(0f, xB - xA) * max(0f, yB - yA)
        val boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top)
        val boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top)

        return interArea / (boxAArea + boxBArea - interArea)
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("best_float16.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabelList(): List<String> {
        return assets.open("labels.txt").bufferedReader().useLines { it.toList() }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            get_permission()
        }
    }
}