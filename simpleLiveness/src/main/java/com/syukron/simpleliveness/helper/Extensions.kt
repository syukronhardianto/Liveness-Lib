package com.simple.livenesslib.helper

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.Rect
import com.google.mlkit.vision.face.Face
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import kotlin.math.sqrt

fun newBoundingBox(face: Face): Rect {
    val boundingBox = face.boundingBox

    val paddingFactor = 0.16
    val newWidth = ((boundingBox.width() - 50) * (1 + paddingFactor)).toInt()
    val newHeight = (boundingBox.height() * (1 + paddingFactor)).toInt()
    val xDiff = (newWidth - boundingBox.width()) / 2
    val yDiff = (newHeight - boundingBox.height()) / 2

    return Rect(
        boundingBox.left - xDiff,
        boundingBox.top - yDiff,
        boundingBox.right + xDiff,
        boundingBox.bottom + yDiff
    )
}

fun calculateDistance(point1: PointF?, point2: PointF?): Float {
    return if (point1 != null && point2 != null) {
        val deltaX = point2.x - point1.x
        val deltaY = point2.y - point1.y
        sqrt(deltaX * deltaX + deltaY * deltaY)
    } else {
        0.0f
    }
}

fun performTextureAnalysis(faceBitmap: Bitmap): Boolean {
    val grayImage = Mat()
    Utils.bitmapToMat(faceBitmap, grayImage)
    Imgproc.cvtColor(grayImage, grayImage, Imgproc.COLOR_BGR2GRAY)

    val edges = Mat()
    Imgproc.Canny(grayImage, edges, 50.0, 150.0)

    val thresholdValue = 2500
    return Core.countNonZero(edges) > thresholdValue
}

fun performMaskAndGlassesDetection(
    bitmap: Bitmap,
    accessoriesNetImageProcessor: ImageProcessor,
    accessoriesNetModelInterpreter: Interpreter
): Boolean {
    val tensorImage = TensorImage.fromBitmap(bitmap)
    val faceNetByteBuffer = accessoriesNetImageProcessor.process(tensorImage).buffer
    val outputArray = Array(1) { FloatArray(NUM_CLASSES) }
    accessoriesNetModelInterpreter.run(faceNetByteBuffer, outputArray)

    val confidencePlainFace = outputArray[0][PLAIN_CLASS_INDEX]
    return confidencePlainFace > 0.55
}

private const val PLAIN_CLASS_INDEX = 3
private const val NUM_CLASSES = 4