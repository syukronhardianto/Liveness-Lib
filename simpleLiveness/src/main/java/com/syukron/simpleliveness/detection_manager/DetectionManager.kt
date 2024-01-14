package com.simple.livenesslib.detection_manager

import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import android.view.Surface.ROTATION_0
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.*
import com.simple.livenesslib.helper.*
import com.syukron.simpleliveness.camera_manager.CameraManager
import org.opencv.android.OpenCVLoader
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class DetectionManager(
    private val processCameraProvider: ProcessCameraProvider,
    private val cameraSelector: CameraSelector,
    private val faceDetectionListener: CameraManager.FaceDetectionListener
) {
    private lateinit var detector: FaceDetector
    private lateinit var imageAnalysis: ImageAnalysis
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var lifecycleOwner: LifecycleOwner
    private lateinit var faceBitmap: Bitmap
    private lateinit var croppedFaceBitmap: Bitmap
    private lateinit var accessoriesNetModelInterpreter: Interpreter
    private lateinit var accessoriesNetImageProcessor: ImageProcessor

    private var scanCounter = 0

    private var faceInDistance = false
    private var hasTexture = false
    private var noAccessories = false

    fun initializeMlkit(lifecycleOwner: LifecycleOwner, context: Context) {
        this.lifecycleOwner = lifecycleOwner
        OpenCVLoader.initDebug()
        accessoriesNetModelInterpreter = Interpreter(FileUtil.loadMappedFile(context, "saved_model.tflite"), Interpreter.Options())
        accessoriesNetImageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
        bindInputAnalyzer()
    }

    private fun bindInputAnalyzer() {
        detector = FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .setMinFaceSize(100f)
                .build()
        )

        imageAnalysis = ImageAnalysis.Builder()
            .setTargetRotation(ROTATION_0)
            .build()

        cameraExecutor = Executors.newSingleThreadExecutor()

        imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
            processImageProxy(detector, imageProxy)
        }

        try {
            processCameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, imageAnalysis)
        } catch (illegalStateException: IllegalStateException) {
            Log.e(ContentValues.TAG, illegalStateException.message ?: "IllegalStateException")
        } catch (illegalArgumentException: IllegalArgumentException) {
            Log.e(ContentValues.TAG, illegalArgumentException.message ?: "IllegalArgumentException")
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun processImageProxy(detector: FaceDetector, imageProxy: ImageProxy) {
        val image = imageProxy.image
        image?.let { img ->
            val inputImage = InputImage.fromMediaImage(img, imageProxy.imageInfo.rotationDegrees)
            detector.process(inputImage).addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    faces.forEach { face ->
                        scanCounter++
                        faceBitmap = BitmapUtils.getBitmap(imageProxy)!!
                        croppedFaceBitmap = BitmapUtils.cropBitmap(faceBitmap, newBoundingBox(face))

                        val rotX = face.headEulerAngleX
                        val rotY = face.headEulerAngleY
                        val leftEye = face.rightEyeOpenProbability ?: 0f
                        val rightEye = face.leftEyeOpenProbability ?: 0f
                        val upperLip = face.getContour(FaceContour.UPPER_LIP_TOP)?.points ?: emptyList()
                        val lowerLip = face.getContour(FaceContour.LOWER_LIP_BOTTOM)?.points ?: emptyList()
                        val smiling = face.smilingProbability ?: 0f
                        val leftEyeMark = face.getLandmark(FaceLandmark.LEFT_EYE)
                        val rightEyeMark = face.getLandmark(FaceLandmark.RIGHT_EYE)
                        faceInDistance = calculateDistance(leftEyeMark?.position, rightEyeMark?.position) > 85.0
                        if (scanCounter % 8 == 0) {
                            hasTexture = performTextureAnalysis(croppedFaceBitmap)
                            noAccessories = performMaskAndGlassesDetection(croppedFaceBitmap, accessoriesNetImageProcessor, accessoriesNetModelInterpreter)
                        }

                        faceDetectionListener.onFaceValidation(
                            true,
                            faceInDistance,
                            hasTexture,
                            noAccessories
                        )

                        faceDetectionListener.onFaceDetection(
                            rotX,
                            rotY,
                            leftEye,
                            rightEye,
                            upperLip,
                            lowerLip,
                            smiling
                        )
                    }
                } else {
                    faceDetectionListener.onFaceValidation(
                        false,
                        faceInDistance,
                        hasTexture,
                        noAccessories
                    )
                }
            }.addOnFailureListener { exception ->
                exception.printStackTrace()
            }.addOnCompleteListener {
                imageProxy.close()
            }
        }
    }

    fun close() {
        detector.close()
        cameraExecutor.shutdown()
    }
}