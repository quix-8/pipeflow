package com.quix.pipeflow

import java.awt.Color
import java.awt.Font
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import kotlin.test.Test
import kotlin.test.assertTrue

class TesseractEngineTest {
    private fun render(text: String): BufferedImage {
        val image = BufferedImage(900, 160, BufferedImage.TYPE_INT_RGB)
        val g = image.createGraphics()
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)
        g.color = Color.WHITE
        g.fillRect(0, 0, image.width, image.height)
        g.color = Color.BLACK
        g.font = Font(Font.SANS_SERIF, Font.PLAIN, 64)
        g.drawString(text, 30, 100)
        g.dispose()
        return image
    }

    @Test
    fun recognizesEnglishAndRussian() {
        val result = OCRCombiner(TesseractEngine()).run(ImageBlock(render("Hello Привет 12345"), ""))
        println(result)
        assertTrue("Hello" in result.fullText, result.fullText)
        assertTrue("Привет" in result.fullText, result.fullText)
        assertTrue("12345" in result.fullText, result.fullText)
    }
}
