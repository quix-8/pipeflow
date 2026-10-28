package com.quix.pipeflow

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeContentPadding
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.quix.pipeflow.modules.LocalServer
import com.quix.pipeflow.modules.ServerState
import com.quix.pipeflow.modules.PipelineKind
import com.quix.pipeflow.modules.RecognizeResponse
import io.github.vinceglb.filekit.PlatformFile
import io.github.vinceglb.filekit.name
import io.github.vinceglb.filekit.dialogs.FileKitType
import io.github.vinceglb.filekit.dialogs.compose.rememberFilePickerLauncher
import io.github.vinceglb.filekit.readBytes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.awt.BasicStroke
import java.awt.Color
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO
import kotlin.io.encoding.Base64

private class SelectedImage(val name: String, val bytes: ByteArray, val image: BufferedImage)

private suspend fun PlatformFile.load(): SelectedImage? = withContext(Dispatchers.IO) {
    val bytes = readBytes()
    ImageIO.read(ByteArrayInputStream(bytes))?.let { SelectedImage(name, bytes, it) }
}

private fun annotate(result: RecognizeResponse): BufferedImage {
    val prepared = ImageIO.read(ByteArrayInputStream(Base64.decode(result.preparedPng)))
    val w = prepared.width
    val h = prepared.height
    val out = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    out.createGraphics().apply {
        setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
        drawImage(prepared, 0, 0, null)
        val stroke = (maxOf(w, h) / 300f).coerceAtLeast(2f)
        this.stroke = BasicStroke(stroke)
        font = font.deriveFont(stroke * 10f)
        for (rec in result.recognitions) {
            val b = rec.region.bounds
            color = Color(220, 30, 30)
            drawRect(b.x, b.y, b.width, b.height)
            if (rec.text.length <= 4) drawString(rec.text, b.x, (b.y - stroke * 2).toInt().coerceAtLeast(font.size))
        }
        dispose()
    }
    return out
}

@Composable
@Preview
fun App() {
    MaterialTheme {
        var kind by remember { mutableStateOf(PipelineKind.MNIST) }
        val server = remember { LocalServer() }
        LaunchedEffect(server) { server.start() }
        DisposableEffect(server) { onDispose { server.close() } }
        val serverState by server.state.collectAsState()
        val client = (serverState as? ServerState.Ready)?.client
        val scope = rememberCoroutineScope()
        var source by remember { mutableStateOf<SelectedImage?>(null) }
        var annotated by remember { mutableStateOf<BufferedImage?>(null) }
        var result by remember { mutableStateOf<RecognizeResponse?>(null) }
        var error by remember { mutableStateOf<String?>(null) }
        var running by remember { mutableStateOf(false) }

        val launcher = rememberFilePickerLauncher(
            type = FileKitType.Image,
            onError = { error = it.message ?: "Не удалось открыть файл" },
            onResult = { file ->
                scope.launch {
                    error = null
                    result = null
                    annotated = null
                    source = file?.load()
                }
            },
        )

        Column(
            modifier = Modifier
                .safeContentPadding()
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                PipelineKind.entries.forEach { option ->
                    RadioButton(
                        selected = kind == option,
                        onClick = { kind = option; result = null; annotated = null },
                        enabled = !running,
                    )
                    Text(option.label)
                    Spacer(Modifier.width(16.dp))
                }
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = androidx.compose.foundation.layout.Arrangement.spacedBy(12.dp)) {
                Button(onClick = { launcher.launch() }, enabled = !running) {
                    Text("Pick an image")
                }
                Button(
                    enabled = client != null && source != null && !running,
                    onClick = {
                        val selected = source ?: return@Button
                        val ocr = client ?: return@Button
                        scope.launch {
                            running = true
                            error = null
                            try {
                                val res = ocr.recognize(selected.bytes, selected.name, kind)
                                val picture = withContext(Dispatchers.Default) { annotate(res) }
                                result = res
                                annotated = picture
                            } catch (e: Exception) {
                                error = e.message ?: e::class.simpleName
                            } finally {
                                running = false
                            }
                        }
                    },
                ) {
                    Text("Recognize")
                }
            }
            Spacer(Modifier.height(16.dp))

            when (val state = serverState) {
                is ServerState.Starting -> Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(state.message)
                }
                is ServerState.Failed -> {
                    Text("Сервер не запустился: ${state.message}", color = MaterialTheme.colorScheme.error)
                    Button(onClick = { scope.launch { server.start() } }) { Text("Повторить") }
                }
                is ServerState.Ready -> {}
            }
            if (running) CircularProgressIndicator()
            error?.let { Text("Ошибка: $it", color = MaterialTheme.colorScheme.error) }

            val shown = annotated ?: source?.image
            if (shown != null) {
                Image(
                    bitmap = shown.toComposeImageBitmap(),
                    contentDescription = null,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxWidth().height(400.dp),
                )
            }

            Spacer(Modifier.height(16.dp))
            result?.let { res ->
                if (res.blocks.isEmpty()) {
                    Text("Цифры не найдены")
                } else {
                    res.blocks.forEach { block ->
                        Text(block.text, style = MaterialTheme.typography.headlineMedium)
                        Text("confidence: ${"%.1f".format(block.confidence * 100)}%")
                    }
                    Spacer(Modifier.height(8.dp))
                    Text("${res.processingTimeMs} ms", style = MaterialTheme.typography.bodySmall)
                    Text(
                        res.recognitions.joinToString("  ") { "${it.text} (${"%.0f".format(it.confidence * 100)}%)" },
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}
