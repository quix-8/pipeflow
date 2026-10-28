package com.quix.pipeflow.modules

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
