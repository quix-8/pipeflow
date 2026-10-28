package com.quix.pipeflow.modules

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.File
import javax.imageio.ImageIO
import org.opencv.core.Rect as CvRect

object OpenCv {
    private val loaded by lazy { nu.pattern.OpenCV.loadLocally() }

    fun ensureLoaded() {
        loaded
    }
}

// Ext функция для преобразование изображения в формат opencv матрицы
internal fun BufferedImage.toMat(): Mat {
    OpenCv.ensureLoaded()
    val bgr = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
    bgr.createGraphics().apply {
        color = Color.WHITE
        fillRect(0, 0, width, height)
        drawImage(this@toMat, 0, 0, null)
        dispose()
    }
    return Mat(height, width, CvType.CV_8UC3).also {
        it.put(0, 0, (bgr.raster.dataBuffer as DataBufferByte).data)
    }
}

// Ext функция для преобразование opencv матрицы в java изображение
internal fun Mat.toBufferedImage(): BufferedImage {
    require(type() == CvType.CV_8UC1 || type() == CvType.CV_8UC3) { "Unsupported Mat type: ${type()}" }
    val gray = type() == CvType.CV_8UC1
    val image = BufferedImage(cols(), rows(), if (gray) BufferedImage.TYPE_BYTE_GRAY else BufferedImage.TYPE_3BYTE_BGR)
    get(0, 0, (image.raster.dataBuffer as DataBufferByte).data)
    return image
}

// Обертка для использования изображения с автоочисткой памяти из c++
private inline fun BufferedImage.useMat(block: (Mat) -> Mat): BufferedImage {
    val src = toMat()
    val dst = block(src)
    return try {
        dst.toBufferedImage()
    } finally {
        src.release()
        dst.release()
    }
}

// Обесцвечивание
class GrayscaleStep : PreProcessor {
    override fun process(input: BufferedImage): BufferedImage = input.useMat { src ->
        Mat().also { Imgproc.cvtColor(src, it, Imgproc.COLOR_BGR2GRAY) }
    }
}

// Бинаризация в черный и белый по Оцу для сканов
class BinarizeStep : PreProcessor {
    override fun process(input: BufferedImage): BufferedImage = input.useMat { src ->
        val gray = if (src.channels() == 1) src else Mat().also { Imgproc.cvtColor(src, it, Imgproc.COLOR_BGR2GRAY) }
        val dst = Mat()
        Imgproc.threshold(gray, dst, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)

        if (Core.mean(dst).`val`[0] > 127) Core.bitwise_not(dst, dst)
        if (gray !== src) gray.release()
        dst
    }
}

// Оцентровка цифры
class CenterDigitStep(
    private val size: Int = 28,
    private val box: Int = 20,
) : PreProcessor {
    override fun process(input: BufferedImage): BufferedImage = input.useMat { src ->
        val gray = if (src.channels() == 1) src else Mat().also { Imgproc.cvtColor(src, it, Imgproc.COLOR_BGR2GRAY) }
        val canvas = Mat.zeros(size, size, CvType.CV_8UC1)

        val points = Mat()
        Core.findNonZero(gray, points)
        if (!points.empty()) {
            val bounds = Imgproc.boundingRect(points)
            val digit = Mat(gray, bounds)
            val scale = box.toDouble() / maxOf(bounds.width, bounds.height)
            val w = (bounds.width * scale).toInt().coerceAtLeast(1)
            val h = (bounds.height * scale).toInt().coerceAtLeast(1)
            val resized = Mat()
            Imgproc.resize(digit, resized, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            resized.copyTo(Mat(canvas, CvRect((size - w) / 2, (size - h) / 2, w, h)))
            resized.release()
            canvas.shiftToCenterOfMass()
        }
        points.release()
        if (gray !== src) gray.release()
        canvas
    }

    private fun Mat.shiftToCenterOfMass() {
        val m = Imgproc.moments(this, false)
        if (m.m00 == 0.0) return
        val dx = Math.round(cols() / 2.0 - m.m10 / m.m00).toDouble()
        val dy = Math.round(rows() / 2.0 - m.m01 / m.m00).toDouble()
        val shift = Mat(2, 3, CvType.CV_64F)
        shift.put(0, 0, 1.0, 0.0, dx, 0.0, 1.0, dy)
        val shifted = Mat()
        Imgproc.warpAffine(this, shifted, shift, size())
        shifted.copyTo(this)
        shifted.release()
        shift.release()
    }
}

// Ресайз под квадрат
class ResizeStep(private val maxSide: Int = 1000) : PreProcessor {
    override fun process(input: BufferedImage): BufferedImage {
        val side = maxOf(input.width, input.height)
        if (side <= maxSide) return input
        val scale = maxSide.toDouble() / side
        return input.useMat { src ->
            Mat().also {
                Imgproc.resize(src, it, Size(input.width * scale, input.height * scale), 0.0, 0.0, Imgproc.INTER_AREA)
            }
        }
    }
}

// Адаптивная бинаризация для фоток с размытием и удалением неоднородностей фона
class AdaptiveBinarizeStep(
    private val windowRatio: Double = 0.25,
    private val offset: Double = 10.0,
) : PreProcessor {
    override fun process(input: BufferedImage): BufferedImage = input.useMat { src ->
        val gray = if (src.channels() == 1) src.clone() else Mat().also { Imgproc.cvtColor(src, it, Imgproc.COLOR_BGR2GRAY) }
        if (Core.mean(gray).`val`[0] < 100) Core.bitwise_not(gray, gray)
        Imgproc.GaussianBlur(gray, gray, Size(5.0, 5.0), 0.0)
        val window = (minOf(gray.rows(), gray.cols()) * windowRatio).toInt().coerceAtLeast(15) or 1
        val dst = Mat()
        Imgproc.adaptiveThreshold(
            gray, dst, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, window, offset,
        )
        gray.release()
        dst
    }
}

class ExtractDigitStep(
    private val minAreaRatio: Double = 0.0002,
    private val partAreaRatio: Double = 0.1,
    private val padding: Double = 0.1,
) : PreProcessor {
    private class Component(val index: Int, val x: Int, val y: Int, val w: Int, val h: Int, val area: Int)

    override fun process(input: BufferedImage): BufferedImage = input.useMat { src ->
        val bin = if (src.channels() == 1) src.clone() else Mat().also { Imgproc.cvtColor(src, it, Imgproc.COLOR_BGR2GRAY) }
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(bin, labels, stats, centroids, 8, CvType.CV_32S)

        val all = (1 until count).map {
            fun stat(i: Int) = stats.get(it, i)[0].toInt()
            Component(
                it, stat(Imgproc.CC_STAT_LEFT), stat(Imgproc.CC_STAT_TOP),
                stat(Imgproc.CC_STAT_WIDTH), stat(Imgproc.CC_STAT_HEIGHT), stat(Imgproc.CC_STAT_AREA),
            )
        }
        val minArea = maxOf(20.0, bin.rows() * bin.cols() * minAreaRatio)
        val sized = all.filter { it.area >= minArea }
        // компоненты у границы кадра — это края листа и тени, а не цифра
        val inner = sized.filter { it.x > 0 && it.y > 0 && it.x + it.w < bin.cols() && it.y + it.h < bin.rows() }
        val pool = inner.ifEmpty { sized }

        val result = if (pool.isEmpty()) {
            bin.clone()
        } else {
            val main = pool.maxBy { it.area }
            val margin = maxOf(main.w, main.h) / 2
            val kept = pool.filter {
                it === main || (
                    it.area >= main.area * partAreaRatio &&
                        it.x + it.w / 2 in (main.x - margin)..(main.x + main.w + margin) &&
                        it.y + it.h / 2 in (main.y - margin)..(main.y + main.h + margin)
                    )
            }
            val mask = Mat.zeros(bin.size(), CvType.CV_8UC1)
            val part = Mat()
            kept.forEach {
                Core.compare(labels, Scalar(it.index.toDouble()), part, Core.CMP_EQ)
                Core.bitwise_or(mask, part, mask)
            }
            part.release()

            val left = kept.minOf { it.x }
            val top = kept.minOf { it.y }
            val w = kept.maxOf { it.x + it.w } - left
            val h = kept.maxOf { it.y + it.h } - top
            val side = (maxOf(w, h) * (1 + 2 * padding)).toInt()
            val square = Mat.zeros(side, side, CvType.CV_8UC1)
            Mat(mask, CvRect(left, top, w, h)).copyTo(Mat(square, CvRect((side - w) / 2, (side - h) / 2, w, h)))
            mask.release()
            square
        }
        bin.release()
        labels.release()
        stats.release()
        centroids.release()
        result
    }
}

// Увеличение четкости цифры для фотографий
class ThickenStep(private val ratio: Double = 0.06) : PreProcessor {
    override fun process(input: BufferedImage): BufferedImage = input.useMat { src ->
        val gray = if (src.channels() == 1) src else Mat().also { Imgproc.cvtColor(src, it, Imgproc.COLOR_BGR2GRAY) }
        val k = (maxOf(gray.rows(), gray.cols()) * ratio).toInt().coerceAtLeast(1) or 1
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
        val dst = Mat()
        Imgproc.dilate(gray, dst, kernel)
        kernel.release()
        if (gray !== src) gray.release()
        dst
    }
}

class SaveImageStep(private val path: String) : PreProcessor {
    override fun process(input: BufferedImage): BufferedImage {
        File(path).apply { parentFile?.mkdirs() }.let { ImageIO.write(input, "png", it) }
        return input
    }
}

fun mnistPreProcessor(debugDir: String? = null): PreProcessor {
    val steps = listOf(
        "1_resize" to ResizeStep(),
        "2_gray" to GrayscaleStep(),
        "3_binary" to AdaptiveBinarizeStep(),
        "4_digit" to ExtractDigitStep(),
        "5_thick" to ThickenStep(),
        "6_mnist" to CenterDigitStep(),
    )
    return PreProcessorChain(
        steps.flatMap { (name, step) ->
            if (debugDir == null) listOf(step) else listOf(step, SaveImageStep("$debugDir/$name.png"))
        },
    )
}
