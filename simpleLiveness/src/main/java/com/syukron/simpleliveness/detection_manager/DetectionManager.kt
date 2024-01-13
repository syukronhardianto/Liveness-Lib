package com.simple.livenesslib.detection_manager

import android.annotation.SuppressLint
import android.content.ContentValues
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
import com.simple.livenesslib.helper.BitmapUtils
import com.simple.livenesslib.helper.calculateDistance
import com.simple.livenesslib.helper.newBoundingBox
import com.simple.livenesslib.helper.performTextureAnalysis
import com.syukron.simpleliveness.camera_manager.CameraManager
import org.opencv.android.OpenCVLoader
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

    private var scanCounter = 0

    private var faceInDistance = false
    private var hasTexture = false

    fun initializeMlkit(lifecycleOwner: LifecycleOwner) {
        this.lifecycleOwner = lifecycleOwner
        OpenCVLoader.initDebug()
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
                        if (scanCounter % 8 == 0) hasTexture = performTextureAnalysis(croppedFaceBitmap)

                        faceDetectionListener.onFaceValidation(
                            true,
                            faceInDistance,
                            hasTexture
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
                        hasTexture
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