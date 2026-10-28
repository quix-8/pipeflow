package com.quix.pipeflow.modules

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