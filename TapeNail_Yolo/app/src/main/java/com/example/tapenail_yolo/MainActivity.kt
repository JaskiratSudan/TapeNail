// MainActivity.kt
package com.example.tapenail_yolo

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.*
import android.media.RingtoneManager
import android.os.*
import android.util.Log
import android.view.Surface
import android.graphics.SurfaceTexture
import android.view.TextureView
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {

    // --- Views & UI ---
    private lateinit var imageView: ImageView
    private lateinit var lockIcon: ImageView
    private lateinit var lockStatus: TextView
    private lateinit var unlockMessage: TextView
    private lateinit var resetButton: ImageButton
    private lateinit var textureView: TextureView

    // new UI
    private lateinit var patternSpinner: Spinner
    private lateinit var latencyText: TextView
    private lateinit var bitmap: Bitmap


    // --- Camera & threading ---
    private lateinit var cameraDevice: CameraDevice
    private lateinit var handler: Handler
    private lateinit var cameraManager: CameraManager

    // --- TFLite model ---
    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>
    private val MODEL_INPUT_SIZE = 256
    private val MODEL_OUTPUT_NUM_ELEMENTS = 1344
    private val MODEL_NUM_CLASSES = 1
    private val CLASS_COLORS = listOf(
        Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW,
        Color.CYAN, Color.MAGENTA, Color.GRAY
    )
    private val inputBuffer = ByteBuffer
        .allocateDirect(4 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3)
        .order(ByteOrder.nativeOrder())

    // --- Detection state ---
    private var isUnlocked = false
    private var detectionStartTime = 0L
    private val requiredDetectionDuration = 2000L
    private var isDetectionRunning = false

    // --- Unlock pattern selection ---
    private var unlockClassId: Int = 1  // default

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initViews()
        setupCamera()
        setupResetButton()
        initModel()
        checkPermissions()
    }

    private fun initViews() {
        imageView     = findViewById(R.id.imageView)
        lockIcon      = findViewById(R.id.lockIcon)
        lockStatus    = findViewById(R.id.lockStatus)
        unlockMessage = findViewById(R.id.unlockMessage)
        resetButton   = findViewById(R.id.resetButton)
        textureView   = findViewById(R.id.texture)

        patternSpinner = findViewById(R.id.patternSpinner)
        latencyText    = findViewById(R.id.latencyText)

        setupPatternSpinner()
    }

    private fun setupPatternSpinner() {
        // Build a list ["Pattern 0", "Pattern 1", ...]
        val items = (0 until MODEL_NUM_CLASSES).map { "Pattern $it" }
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            items
        ).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        patternSpinner.adapter = adapter
        patternSpinner.setSelection(unlockClassId)
        patternSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>,
                view: View?,
                position: Int,
                id: Long
            ) {
                unlockClassId = position
            }
            override fun onNothingSelected(parent: AdapterView<*>) { /* no-op */ }
        }
    }

    private fun setupResetButton() {
        resetButton.setOnClickListener {
            resetAllDetection()
            unlockMessage.visibility = View.GONE
            lockIcon.visibility       = View.VISIBLE
            lockStatus.visibility     = View.VISIBLE
            openCamera()
        }
    }

    private fun setupCamera() {
        cameraManager = getSystemService(CameraManager::class.java)
        val handlerThread = HandlerThread("videoThread").apply { start() }
        handler = Handler(handlerThread.looper)

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(
                surface: SurfaceTexture, w: Int, h: Int
            ) = openCamera()
            override fun onSurfaceTextureSizeChanged(
                surface: SurfaceTexture, w: Int, h: Int
            ) {}
            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture) = false
            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = processFrame()
        }
    }

    private fun processFrame() {
        if (!isUnlocked) {
            val frame = textureView.bitmap ?: return

            // get the real pixel size of your TextureView / ImageView
            val vw = textureView.width
            val vh = textureView.height

            // make a mutable copy of the camera frame
            val mutable = frame.copy(Bitmap.Config.ARGB_8888, true)

            // scale it to match the view
            val scaled = Bitmap.createScaledBitmap(mutable, vw, vh, true)
            bitmap = scaled

            handler.post {
                val results = runInference(scaled)
                drawDetectionResults(results)
            }
        }
    }


    @SuppressLint("MissingPermission")
    private fun openCamera() {
        try {
            cameraManager.openCamera(
                cameraManager.cameraIdList[0],
                object : CameraDevice.StateCallback() {
                    override fun onOpened(camera: CameraDevice) {
                        cameraDevice = camera
                        val surface = Surface(textureView.surfaceTexture).apply {
                            textureView.surfaceTexture
                                ?.setDefaultBufferSize(640, 640)
                        }
                        cameraDevice.createCaptureSession(
                            listOf(surface),
                            object : CameraCaptureSession.StateCallback() {
                                override fun onConfigured(session: CameraCaptureSession) {
                                    session.setRepeatingRequest(
                                        cameraDevice
                                            .createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                                            .apply { addTarget(surface) }
                                            .build(),
                                        null,
                                        null
                                    )
                                }
                                override fun onConfigureFailed(session: CameraCaptureSession) {
                                    Log.e("MainActivity", "Camera config failed")
                                }
                            },
                            handler
                        )
                    }
                    override fun onDisconnected(camera: CameraDevice) = cleanupCamera()
                    override fun onError(camera: CameraDevice, error: Int) = cleanupCamera()
                },
                handler
            )
        } catch (e: Exception) {
            Log.e("CameraError", "Failed to open camera: ${e.message}")
        }
    }

    private fun drawDetectionResults(results: List<DetectionResult>) {
        runOnUiThread {
            // 1️⃣ Always keep both preview & overlay visible
            textureView.visibility = View.VISIBLE
            imageView.visibility   = View.VISIBLE

            // 2️⃣ Clear overlay when no detections
            if (results.isEmpty()) {
                bitmap.eraseColor(Color.TRANSPARENT)
            }

            // 3️⃣ Draw detections onto the bitmap
            val canvas = Canvas(bitmap)
            var targetDetected = false
            var holdTextPos: Pair<Float, Float>? = null

            results.forEach { result ->
                val paint = Paint().apply {
                    color       = CLASS_COLORS[result.classId % CLASS_COLORS.size]
                    style       = Paint.Style.STROKE
                    strokeWidth = 5f
                    textSize    = 40f
                }

                val rect = RectF(
                    result.xmin * bitmap.width,
                    result.ymin * bitmap.height,
                    result.xmax * bitmap.width,
                    result.ymax * bitmap.height
                )
                canvas.drawRect(rect, paint)
                canvas.drawText(
                    "${labels.getOrElse(result.classId) { "Unknown" }} (${"%.2f".format(result.confidence)})",
                    rect.left,
                    rect.top - 10,
                    paint
                )

                if (result.classId == unlockClassId && result.confidence > 0.6f) {
                    targetDetected = true
                    holdTextPos = result.let {
                        Pair(it.xmin * bitmap.width, it.ymin * bitmap.height - 60)
                    }
                }
            }

            // 4️⃣ Optional “hold” instruction
            holdTextPos?.let { (x, y) ->
                canvas.drawText(
                    "Hold for 2 seconds...",
                    x, y,
                    Paint().apply {
                        color    = Color.WHITE
                        textSize = 50f
                    }
                )
            }

            // 5️⃣ Update detection state (unchanged)
            if (!isUnlocked) {
                handleDetectionState(targetDetected)
            }

            // 6️⃣ Push overlay bitmap to the ImageView
            imageView.setImageBitmap(bitmap)
        }
    }


    private fun handleDetectionState(detected: Boolean) {
        lockStatus.text = if (detected) "Detected - Hold..." else "Locked"
        lockIcon.setColorFilter(if (detected) Color.YELLOW else Color.WHITE)

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
            if (!isDetectionRunning) return@postDelayed

            val elapsed = System.currentTimeMillis() - detectionStartTime
            if (elapsed >= requiredDetectionDuration) {
                // Unlock must happen on UI thread, too
                runOnUiThread { unlockDevice() }
            } else {
                val secondsLeft = 2 - (elapsed / 1000)
                runOnUiThread {
                    lockStatus.text = "Hold for ${secondsLeft}s..."
                    lockIcon.setColorFilter(Color.YELLOW)
                }
                // schedule the next tick
                startUnlockCountdown()
            }
        }, 100)
    }


    private fun unlockDevice() {
        isUnlocked = true
        runOnUiThread {
            RingtoneManager.getRingtone(
                this,
                RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            ).play()
            lockIcon.visibility      = View.GONE
            lockStatus.visibility    = View.GONE
            unlockMessage.visibility = View.VISIBLE
        }
        cleanupCamera()
    }

    private fun cleanupCamera() {
        try { cameraDevice.close() } catch (e: Exception) {
            Log.e("CameraError", "Error closing camera: ${e.message}")
        }
        handler.removeCallbacksAndMessages(null)
    }

    private fun resetAllDetection() {
        isUnlocked = false
        isDetectionRunning = false
        detectionStartTime = 0
        handler.removeCallbacksAndMessages(null)
    }

    private fun resetDetection() {
        isDetectionRunning = false
        handler.removeCallbacksAndMessages(null)
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(
                arrayOf(Manifest.permission.CAMERA),
                101
            )
        }
    }

    private fun initModel() {
        tflite = Interpreter(loadModelFile())
        labels = loadLabelList()
    }

    private data class DetectionResult(
        val xmin: Float,
        val ymin: Float,
        val xmax: Float,
        val ymax: Float,
        val classId: Int,
        val confidence: Float
    )

    private fun runInference(bitmap: Bitmap): List<DetectionResult> {
        val startMs = SystemClock.elapsedRealtime()

        convertBitmapToByteBuffer(bitmap)
        val outputBuffer = Array(1) {
            Array(4 + MODEL_NUM_CLASSES) {
                FloatArray(MODEL_OUTPUT_NUM_ELEMENTS)
            }
        }
        tflite.run(inputBuffer, outputBuffer)

        val results = outputBuffer[0].let { output ->
            (0 until MODEL_OUTPUT_NUM_ELEMENTS).mapNotNull { i ->
                val xCenter = output[0][i]
                val yCenter = output[1][i]
                val width   = output[2][i]
                val height  = output[3][i]
                val scores  = (4 until 4 + MODEL_NUM_CLASSES).map { output[it][i] }
                val maxScore = scores.maxOrNull() ?: 0f
                val clsId    = scores.indexOf(maxScore)

                if (maxScore > 0.5f) DetectionResult(
                    xmin       = (xCenter - width/2).coerceIn(0f,1f),
                    ymin       = (yCenter - height/2).coerceIn(0f,1f),
                    xmax       = (xCenter + width/2).coerceIn(0f,1f),
                    ymax       = (yCenter + height/2).coerceIn(0f,1f),
                    classId    = clsId,
                    confidence = maxScore
                ) else null
            }.let { processResults(it) }
        }

        // update latency display
        val latency = SystemClock.elapsedRealtime() - startMs
        runOnUiThread {
            latencyText.text = "Latency: ${latency} ms"
        }

        return results
    }

    private fun processResults(results: List<DetectionResult>) = results
        .groupBy { it.classId }
        .flatMap { (_, classResults) ->
            applyNMS(
                classResults.map { RectF(it.xmin, it.ymin, it.xmax, it.ymax) },
                classResults.map { it.confidence }
            ).map { classResults[it] }
        }

    private fun applyNMS(
        boxes: List<RectF>,
        scores: List<Float>,
        iouThreshold: Float = 0.5f
    ): List<Int> {
        val selected = mutableListOf<Int>()
        val sorted   = scores.indices.sortedByDescending { scores[it] }
        sorted.forEach { i ->
            if (selected.none { j -> iou(boxes[i], boxes[j]) > iouThreshold }) {
                selected.add(i)
            }
        }
        return selected
    }

    private fun iou(boxA: RectF, boxB: RectF): Float {
        val xA = max(boxA.left, boxB.left)
        val yA = max(boxA.top,  boxB.top)
        val xB = min(boxA.right, boxB.right)
        val yB = min(boxA.bottom,boxB.bottom)

        val interArea = max(0f, xB - xA) * max(0f, yB - yA)
        val areaA = (boxA.right - boxA.left) * (boxA.bottom - boxA.top)
        val areaB = (boxB.right - boxB.left) * (boxB.bottom - boxB.top)
        return interArea / (areaA + areaB - interArea + 1e-6f)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val resized = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        val intVals = IntArray(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE)
        resized.getPixels(
            intVals, 0, MODEL_INPUT_SIZE,
            0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE
        )
        inputBuffer.asFloatBuffer().apply {
            intVals.forEach { pixel ->
                put(((pixel shr 16) and 0xFF) / 255.0f)
                put(((pixel shr 8)  and 0xFF) / 255.0f)
                put((pixel and 0xFF)          / 255.0f)
            }
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        assets.openFd("Gtapenail2_float16.tflite").use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                return fis.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fd.startOffset,
                    fd.declaredLength
                )
            }
        }
    }

    private fun loadLabelList() = assets.open("labels.txt")
        .bufferedReader().useLines { it.toList() }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            openCamera()
        } else {
            checkPermissions()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cleanupCamera()
        tflite.close()
    }
}
