package com.quix.pipeflow

import net.sourceforge.tess4j.ITessAPI
import net.sourceforge.tess4j.Tesseract
import java.awt.Point
import java.awt.image.BufferedImage

interface OCREngine {
    fun recognize(img: ImageBlock): OCRData
}

class TesseractEngine(
    private val dataPath: String = System.getenv("TESSDATA_PREFIX") ?: "/home/quix_/models/tessdata",
    private val language: String = "rus+eng",
) : OCREngine {
    override fun recognize(img: ImageBlock): OCRData {
        val startedAt = System.currentTimeMillis()
        val tesseract = Tesseract().apply {
            setDatapath(dataPath)
            setLanguage(language)
        }
        val blocks = tesseract.getWords(img.image, ITessAPI.TessPageIteratorLevel.RIL_TEXTLINE)
            .filter { it.text.isNotBlank() }
            .map { word ->
                val r = word.boundingBox
                TextBlock(
                    text = word.text.trim(),
                    confidence = word.confidence / 100f,
                    boundingBox = listOf(
                        Point(r.x, r.y),
                        Point(r.x + r.width, r.y),
                        Point(r.x + r.width, r.y + r.height),
                        Point(r.x, r.y + r.height),
                    ),
                )
            }
        return OCRData(
            fullText = blocks.joinToString("\n") { it.text },
            blocks = blocks,
            engineName = "Tesseract ($language)",
            processingTimeMs = System.currentTimeMillis() - startedAt,
        )
    }
}

class ONNXEngine : OCREngine {
    override fun recognize(img: ImageBlock): OCRData {
        TODO("Not yet implemented")
    }
}

class OCRCombiner(
    val engine: OCREngine,
) {
    fun run(img: ImageBlock): OCRData = engine.recognize(img)
}

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
