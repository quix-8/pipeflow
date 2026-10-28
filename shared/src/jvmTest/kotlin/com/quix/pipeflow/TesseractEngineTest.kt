package com.quix.pipeflow

import com.quix.pipeflow.modules.Recognizer
import com.quix.pipeflow.modules.Rect
import com.quix.pipeflow.modules.Region
import com.quix.pipeflow.modules.TesseractEngine
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
        val result = TesseractEngine().recognize(render("Hello Привет 12345"))
        println(result)
        assertTrue("Hello" in result.text, result.text)
        assertTrue("Привет" in result.text, result.text)
        assertTrue("12345" in result.text, result.text)
    }

    @Test
    fun recognizerCropsRegions() {
        val image = render("Hello Привет 12345")
        val region = Region.of(Rect(0, 0, image.width, image.height))
        val result = Recognizer(TesseractEngine()).process(image to listOf(region))
        assertTrue("Hello" in result.single().text, result.single().text)
    }
}
