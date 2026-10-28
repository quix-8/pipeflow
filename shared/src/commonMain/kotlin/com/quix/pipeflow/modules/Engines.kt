package com.quix.pipeflow.modules

import net.sourceforge.tess4j.ITessAPI
import net.sourceforge.tess4j.Tesseract
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.nio.FloatBuffer
import kotlin.math.exp

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

/**
 * Классификатор одной цифры на модели mnist-12.onnx.
 * Ожидает вход после [mnistPreProcessor] (белая цифра на чёрном); изображение другого размера масштабируется до 28×28.
 */
class OnnxMnistEngine(
    modelPath: String = System.getenv("MNIST_MODEL") ?: "/home/quix_/models/mnist-12.onnx",
) : OCREngine, AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelPath, OrtSession.SessionOptions())
    private val inputName = session.inputNames.first()

    override fun recognize(image: BufferedImage): RecognizedText {
        val input = image.toMnistInput()
        val logits = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 1, SIZE.toLong(), SIZE.toLong())).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                (result[0].value as Array<FloatArray>)[0]
            }
        }
        val probabilities = softmax(logits)
        val digit = probabilities.indices.maxBy { probabilities[it] }
        return RecognizedText(digit.toString(), probabilities[digit])
    }

    override fun close() {
        session.close()
    }

    private fun BufferedImage.toMnistInput(): FloatArray {
        val gray = BufferedImage(SIZE, SIZE, BufferedImage.TYPE_BYTE_GRAY)
        gray.createGraphics().apply {
            setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            drawImage(this@toMnistInput, 0, 0, SIZE, SIZE, null)
            dispose()
        }
        return FloatArray(SIZE * SIZE) { gray.raster.getSample(it % SIZE, it / SIZE, 0) / 255f }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exp = FloatArray(logits.size) { exp(logits[it] - max) }
        val sum = exp.sum()
        return FloatArray(exp.size) { exp[it] / sum }
    }

    private companion object {
        const val SIZE = 28
    }
}
