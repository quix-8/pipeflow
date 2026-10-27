package com.quix.pipeflow

import net.sourceforge.tess4j.ITessAPI
import net.sourceforge.tess4j.Tesseract
import java.awt.image.BufferedImage


// Pipeline
interface PipelineStage<in I, out O> {
    fun process(input: I): O
}

interface PreProcessor: PipelineStage<BufferedImage, BufferedImage>

/** Последовательно применяет шаги предобработки; пустая цепочка возвращает изображение без изменений. */
class PreProcessorChain(
    private val steps: List<PreProcessor> = emptyList(),
) : PreProcessor {
    constructor(vararg steps: PreProcessor) : this(steps.toList())

    override fun process(input: BufferedImage): BufferedImage =
        steps.fold(input) { image, step -> step.process(image) }
}

interface Segmentator: PipelineStage<BufferedImage, List<Region>>

interface Assembler: PipelineStage<List<Recognition>, List<TextBlock>>

class Recognizer(
    private val engine: OCREngine,
) : PipelineStage<Pair<BufferedImage, List<Region>>, List<Recognition>> {
    override fun process(input: Pair<BufferedImage, List<Region>>): List<Recognition> {
        val (image, regions) = input
        return regions.map { region ->
            val crop = image.crop(region.bounds)
            val result = if (crop == null) RecognizedText("", 0f) else engine.recognize(crop)
            Recognition(region, result.text, result.confidence)
        }
    }

    private fun BufferedImage.crop(rect: Rect): BufferedImage? {
        val x = rect.x.coerceIn(0, width)
        val y = rect.y.coerceIn(0, height)
        val w = (rect.right.coerceIn(0, width) - x)
        val h = (rect.bottom.coerceIn(0, height) - y)
        if (w <= 0 || h <= 0) return null
        return getSubimage(x, y, w, h)
    }
}

class Pipeline(
    private val preProcessor: PreProcessor,
    private val segmentator: Segmentator,
    private val recognizer: Recognizer,
    private val assembler: Assembler,
) {
    fun run(image: BufferedImage): List<TextBlock> {
        val prepared = preProcessor.process(image)
        val regions = segmentator.process(prepared)
        val recognitions = recognizer.process(prepared to regions)
        return assembler.process(recognitions)
    }
}

// OCR engines
interface OCREngine {
    fun recognize(image: BufferedImage): RecognizedText
}

class TesseractEngine(
    private val dataPath: String = System.getenv("TESSDATA_PREFIX") ?: "/home/quix_/models/tessdata_best",
    private val language: String = "rus+eng",
) : OCREngine {
    override fun recognize(image: BufferedImage): RecognizedText {
        val tesseract = Tesseract().apply {
            setDatapath(dataPath)
            setLanguage(language)
        }
        val lines = tesseract.getWords(image, ITessAPI.TessPageIteratorLevel.RIL_TEXTLINE)
            .filter { it.text.isNotBlank() }
        if (lines.isEmpty()) return RecognizedText("", 0f)
        return RecognizedText(
            text = lines.joinToString("\n") { it.text.trim() },
            confidence = lines.map { it.confidence }.average().toFloat() / 100f,
        )
    }
}

class ONNXEngine : OCREngine {
    override fun recognize(image: BufferedImage): RecognizedText {
        TODO("Not yet implemented")
    }
}

// Data classes
data class Point(val x: Int, val y: Int)

data class Rect(val x: Int, val y: Int, val width: Int, val height: Int) {
    val right: Int get() = x + width
    val bottom: Int get() = y + height

    fun toPolygon(): List<Point> = listOf(
        Point(x, y),
        Point(right, y),
        Point(right, bottom),
        Point(x, bottom),
    )
}

/** Область изображения, найденная сегментатором. [polygon] допускает повёрнутый текст. */
data class Region(
    val polygon: List<Point>,
    val readingOrder: Int = 0,
) {
    init {
        require(polygon.size >= 3) { "Region polygon must have at least 3 points" }
    }

    val bounds: Rect
        get() {
            val minX = polygon.minOf { it.x }
            val minY = polygon.minOf { it.y }
            return Rect(minX, minY, polygon.maxOf { it.x } - minX, polygon.maxOf { it.y } - minY)
        }

    companion object {
        fun of(rect: Rect, readingOrder: Int = 0) = Region(rect.toPolygon(), readingOrder)
    }
}

/** Результат работы движка над одним изображением; confidence в диапазоне 0..1. */
data class RecognizedText(
    val text: String,
    val confidence: Float,
)

/** Результат распознавания одной [region]. */
data class Recognition(
    val region: Region,
    val text: String,
    val confidence: Float,
)

/** Итоговый текстовый блок после сборки (строка/абзац). */
data class TextBlock(
    val text: String,
    val confidence: Float,
    val boundingBox: List<Point>,
)

data class OCRData(
    val fullText: String,
    val blocks: List<TextBlock> = emptyList(),
    val engineName: String = "",
    val processingTimeMs: Long = 0,
)

data class ImageBlock(
    var image: BufferedImage,
    var path: String,
)
