package com.simple.livenesslib.detection_manager

import android.graphics.PointF
import android.util.Log

class DetectionCategories {
    private var faceUpLow = false
    private var faceUpHi = false
    private var faceDownLow = false
    private var faceDownHi = false
    private var faceRightLow = false
    private var faceRightHi = false
    private var faceLeftLow = false
    private var faceLeftHi = false
    private var eyesClose = false
    private var eyesOpen = false
    private var openMouth = false
    private var smilingProb = false

    fun initializeValue() {
        faceUpLow = false
        faceUpHi = false
        faceDownLow = false
        faceDownHi = false
        faceRightLow = false
        faceRightHi = false
        faceLeftLow = false
        faceLeftHi = false
        eyesClose = false
        eyesOpen = false
        openMouth = false
        smilingProb = false
    }

    fun faceValidation(face: Boolean): Boolean {
        return face
    }

    fun detectFaceUp(vertical: Float): Boolean {
        if (vertical in -10.0..10.0) faceUpLow = true
        if (faceUpLow && vertical > 15.0) faceUpHi = true
        return faceUpLow && faceUpHi
    }

    fun detectFaceDown(vertical: Float): Boolean {
        Log.d("my_vertical", vertical.toString())
        if (vertical in -10.0..10.0) faceDownLow = true
        if (faceDownLow && vertical < -22.0) faceDownHi = true
        return faceDownLow && faceDownHi
    }

    fun detectFaceRight(horizontal: Float): Boolean {
        if (horizontal in -10.0..10.0) faceRightLow = true
        if (faceRightLow && horizontal < -33.0) faceRightHi = true
        return faceRightLow && faceRightHi
    }

    fun detectFaceLeft(horizontal: Float): Boolean {
        if (horizontal in -10.0..10.0) faceLeftLow = true
        if (faceLeftLow && horizontal > 30.0) faceLeftHi = true
        return faceLeftLow && faceLeftHi
    }

    fun detectBlinkEyes(leftEye: Float, rightEye: Float): Boolean {
        if (leftEye < 0.1 && rightEye < 0.1) eyesClose = true
        if (leftEye > 0.4 && rightEye > 0.4) eyesOpen = true
        return eyesClose && eyesOpen
    }

    fun detectMouthOpened(upperLip: List<PointF>?, lowerLip: List<PointF>?): Boolean {
        val mouthCalculated = calculateMouthOpening(upperLip, lowerLip)
        if (mouthCalculated > 8) openMouth = true
        return openMouth
    }

    private fun calculateMouthOpening(upperLip: List<PointF>?, lowerLip: List<PointF>?): Float {
        if (upperLip != null && lowerLip != null && upperLip.isNotEmpty() && lowerLip.isNotEmpty()) {
            val upperLipCenter = PointF((upperLip[0].x + upperLip[upperLip.size - 1].x) / 2,
                (upperLip[0].y + upperLip[upperLip.size - 1].y) / 2)

            val lowerLipCenter = PointF((lowerLip[0].x + lowerLip[lowerLip.size - 1].x) / 2,
                (lowerLip[0].y + lowerLip[lowerLip.size - 1].y) / 2)

            return lowerLipCenter.y - upperLipCenter.y
        }
        return 0f
    }

    fun detectSmiling(smiling: Float): Boolean {
        if (smiling > 0.7) smilingProb = true
        return smilingProb
    }
}