package com.quix.pipeflow.modules

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class Point(val x: Int, val y: Int)

data class Rect(val x: Int, val y: Int, val width: Int, val height: Int) {
    val right: Int get() = x + width
    val bottom: Int get() = y + height
}

/** Область изображения, найденная сегментатором на сервере. [polygon] допускает повёрнутый текст. */
@Serializable
data class Region(
    val polygon: List<Point>,
    val readingOrder: Int = 0,
    val row: Int = 0,
    val column: Int = 0,
) {
    val bounds: Rect
        get() {
            val minX = polygon.minOf { it.x }
            val minY = polygon.minOf { it.y }
            return Rect(minX, minY, polygon.maxOf { it.x } - minX, polygon.maxOf { it.y } - minY)
        }
}

/** Результат распознавания одной [region]; confidence в диапазоне 0..1. */
@Serializable
data class Recognition(
    val region: Region,
    val text: String,
    val confidence: Float,
)

/** Итоговый текстовый блок после сборки (строка/абзац). */
@Serializable
data class TextBlock(
    val text: String,
    val confidence: Float,
    val boundingBox: List<Point>,
)

@Serializable
enum class PipelineKind(val label: String) {
    @SerialName("mnist")
    MNIST("MNIST (цифры)"),

    @SerialName("mnist_table")
    MNIST_TABLE("MNIST (таблица)"),

    @SerialName("tesseract")
    TESSERACT("Tesseract (текст)"),
}

/** Ответ сервера; координаты регионов относятся к подготовленному изображению [preparedPng] (PNG в base64). */
@Serializable
data class RecognizeResponse(
    val pipeline: PipelineKind,
    val preparedPng: String,
    val width: Int,
    val height: Int,
    val recognitions: List<Recognition>,
    val blocks: List<TextBlock>,
    val processingTimeMs: Long,
)
