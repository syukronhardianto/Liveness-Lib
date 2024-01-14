package com.syukron.simpleliveness.camera_manager

import android.content.ContentValues
import android.content.Context
import android.graphics.PointF
import android.util.Log
import android.view.Surface.ROTATION_0
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import com.simple.livenesslib.detection_manager.DetectionManager

class CameraManager(
    private val context: Context,
    private val surfaceProvider: Preview.SurfaceProvider,
    private val faceDetectionListener: FaceDetectionListener
) {
    interface FaceDetectionListener {
        fun onFaceDetection(
            vertical: Float,
            horizontal: Float,
            leftEye: Float,
            rightEye: Float,
            upperLip: List<PointF>,
            lowerLip: List<PointF>,
            smiling: Float
        )

        fun onFaceValidation(
            faceDetected: Boolean,
            faceInDistance: Boolean,
            hasTexture: Boolean,
            noAccessories: Boolean
        )
    }

    private lateinit var cameraSelector: CameraSelector
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private lateinit var processCameraProvider: ProcessCameraProvider
    private lateinit var cameraPreview: Preview
    private lateinit var lifecycleOwner: LifecycleOwner
    private lateinit var detectionManager: DetectionManager

    fun initializeCamera(lifecycleOwner: LifecycleOwner) {
        this.lifecycleOwner = lifecycleOwner
        startCamera()
    }

    private fun startCamera() {
        cameraSelector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_FRONT).build()
        cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener(
            {
                processCameraProvider = cameraProviderFuture.get()
                initializeCameraPreview()
            }, ContextCompat.getMainExecutor(context)
        )
    }

    private fun initializeCameraPreview() {
        cameraPreview = createCameraPreview()
        bindCameraPreview()
        detectionManager = DetectionManager(processCameraProvider, cameraSelector, faceDetectionListener)
        detectionManager.initializeMlkit(lifecycleOwner, context)
    }

    private fun createCameraPreview(): Preview {
        return Preview.Builder()
            .setTargetRotation(ROTATION_0)
            .build()
    }

    private fun bindCameraPreview() {
        cameraPreview.setSurfaceProvider(surfaceProvider)
        try {
            processCameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, cameraPreview)
        } catch (illegalStateException: IllegalStateException) {
            Log.e(ContentValues.TAG, illegalStateException.message ?: "IllegalStateException")
        } catch (illegalArgumentException: IllegalArgumentException) {
            Log.e(ContentValues.TAG, illegalArgumentException.message ?: "IllegalArgumentException")
        }
    }

    fun closeDetectionManager() {
        detectionManager.close()
    }
}