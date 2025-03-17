package com.example.tapenail_yolo

import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.content.res.Resources
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
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.os.Looper
import android.util.Log
import android.view.Gravity
import android.widget.RelativeLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import com.airbnb.lottie.LottieAnimationView
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

    private var isUnlocked = false
    private var detectionStartTime: Long = 0
    private val requiredDetectionDuration = 5000L // 5 seconds
    //    private val handler = Handler(Looper.getMainLooper())
    private lateinit var mediaPlayer: MediaPlayer
    private var isDetectionRunning = false

    private val MODEL_INPUT_WIDTH = 256
    private val MODEL_INPUT_HEIGHT = 256
    private val MODEL_OUTPUT_NUM_ELEMENTS = 1344
    private val MODEL_NUM_CLASSES = 7 // Update this based on your labels
    private val CLASS_COLORS = listOf(
        Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW,
        Color.CYAN, Color.MAGENTA, Color.GRAY, Color.WHITE,
        Color.rgb(255, 165, 0) // Orange
    )

// Original (for 256x256)
// private val inputBuffer = ByteBuffer.allocateDirect(4 * 256 * 256 * 3)

    // New (for 640x640)
    // Update buffer initialization
    private val inputBuffer = ByteBuffer.allocateDirect(4 * MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3).order(ByteOrder.nativeOrder())
    private val outputBuffer = Array(1) { Array(4 + MODEL_NUM_CLASSES) { FloatArray(MODEL_OUTPUT_NUM_ELEMENTS) } }
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
                if (!isUnlocked) {
                    bitmap = textureView.bitmap!!
                    handler.post {
                        val croppedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
                        val results = runInference(croppedBitmap)
                        drawDetectionResults(results)
                    }
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
            var targetClassDetected = false

            for (result in results) {
                // Draw bounding boxes as before

                // Check if target class (e.g., class 0) is detected
                if (result.classId == 6 && result.confidence > 0.7) {
                    targetClassDetected = true
                }
            }

            if (!isUnlocked) {
                handlePatternDetection(targetClassDetected)
            }

            imageView.setImageBitmap(bitmap)
        }
    }

    private fun handlePatternDetection(detected: Boolean) {
        if (detected) {
            if (!isDetectionRunning) {
                detectionStartTime = System.currentTimeMillis()
                isDetectionRunning = true
                startUnlockCountdown()
            }
        } else {
            resetDetection()
        }
    }

    private fun startUnlockCountdown() {
        handler.postDelayed({
            val elapsedTime = System.currentTimeMillis() - detectionStartTime
            if (elapsedTime >= requiredDetectionDuration && !isUnlocked) {
                unlockPhone()
            } else if (isDetectionRunning) {
                startUnlockCountdown()
            }
        }, 100)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::mediaPlayer.isInitialized) {
            mediaPlayer.release()
        }
        handler.removeCallbacksAndMessages(null)
        if (::cameraDevice.isInitialized) {
            cameraDevice.close()
        }
    }

    private fun resetAllDetection() {
        isUnlocked = false
        isDetectionRunning = false
        detectionStartTime = 0
    }

    private fun resetDetection() {
        isDetectionRunning = false
        handler.removeCallbacksAndMessages(null)
    }

    private fun unlockPhone() {
        isUnlocked = true
        try {
            cameraDevice.close()

            // System default notification sound
            val notification = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            RingtoneManager.getRingtone(this, notification).play()

            // System default animation
            showSystemUnlockAnimation()

            showUnlockedMessage()

        } catch (e: Exception) {
            Log.e("UnlockError", "Error: ${e.message}")
        }
    }

    private fun showSystemUnlockAnimation() {
        runOnUiThread {
            val imageView = ImageView(this).apply {
                setImageResource(android.R.drawable.ic_lock_lock)
                layoutParams = RelativeLayout.LayoutParams(200, 200).apply {
                    addRule(RelativeLayout.CENTER_IN_PARENT)
                }
            }
            findViewById<RelativeLayout>(R.id.main).addView(imageView)
        }
    }

    private fun showUnlockedMessage() {
        runOnUiThread {
            val textView = TextView(this).apply {
                text = "PHONE UNLOCKED"
                textSize = 40f
                setTextColor(Color.GREEN)
                gravity = Gravity.CENTER
            }

            findViewById<RelativeLayout>(R.id.main).addView(textView)
        }
    }

    private fun getColorForClass(classId: Int): Int {
        return CLASS_COLORS[classId % CLASS_COLORS.size]
    }

    // Update bitmap processing
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap,
            MODEL_INPUT_WIDTH,
            MODEL_INPUT_HEIGHT,
            true
        )

        val intValues = IntArray(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT)
        resizedBitmap.getPixels(
            intValues,
            0,
            MODEL_INPUT_WIDTH,
            0,
            0,
            MODEL_INPUT_WIDTH,
            MODEL_INPUT_HEIGHT
        )

        val floatBuffer = inputBuffer.asFloatBuffer()
        for (value in intValues) {
            floatBuffer.put(((value shr 16) and 0xFF) / 255.0f)
            floatBuffer.put(((value shr 8) and 0xFF) / 255.0f)
            floatBuffer.put((value and 0xFF) / 255.0f)
        }
    }

    // Update inference processing
    private fun runInference(bitmap: Bitmap): List<DetectionResult> {
        convertBitmapToByteBuffer(bitmap)
        tflite.run(inputBuffer, outputBuffer)

        val results = mutableListOf<DetectionResult>()
        val output = outputBuffer[0]

        for (i in 0 until MODEL_OUTPUT_NUM_ELEMENTS) {
            val xCenter = output[0][i]
            val yCenter = output[1][i]
            val width = output[2][i]
            val height = output[3][i]

            val classScores = (4 until 4 + MODEL_NUM_CLASSES).map { output[it][i] }
            val maxScore = classScores.maxOrNull() ?: 0f
            val classId = classScores.indexOf(maxScore)

            if (maxScore > 0.5f) {
                results.add(
                    DetectionResult(
                        xmin = (xCenter - width/2).coerceIn(0f, 1f),
                        ymin = (yCenter - height/2).coerceIn(0f, 1f),
                        xmax = (xCenter + width/2).coerceIn(0f, 1f),
                        ymax = (yCenter + height/2).coerceIn(0f, 1f),
                        classId = classId,
                        confidence = maxScore
                    )
                )
            }
        }

        return processResults(results)
    }

    private fun processResults(results: List<DetectionResult>): List<DetectionResult> {
        return results.groupBy { it.classId }
            .flatMap { (_, classResults) ->
                applyNMS(
                    classResults.map { RectF(it.xmin, it.ymin, it.xmax, it.ymax) },
                    classResults.map { it.confidence }
                ).map { classResults[it] }
            }
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
        val fileDescriptor = assets.openFd("multiclass_256_float16.tflite")
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