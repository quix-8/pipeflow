package com.quix.pipeflow

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.safeContentPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import io.github.vinceglb.filekit.PlatformFile
import io.github.vinceglb.filekit.dialogs.FileKitType
import io.github.vinceglb.filekit.dialogs.compose.rememberFilePickerLauncher
import io.github.vinceglb.filekit.readBytes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.jetbrains.compose.resources.painterResource
import pipeflow.shared.generated.resources.Res
import pipeflow.shared.generated.resources.compose_multiplatform
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import javax.imageio.ImageIO

suspend fun PlatformFile.toBufferedImage(): BufferedImage? = withContext(Dispatchers.IO) {
    val bytes = this@toBufferedImage.readBytes()
    ByteArrayInputStream(bytes).use { inputStream ->
        ImageIO.read(inputStream)
    }
}


@Composable
@Preview
fun App() {
    MaterialTheme {
        val scope = rememberCoroutineScope()
        var showError by remember { mutableStateOf(false) }
        var image by remember { mutableStateOf<BufferedImage?>(null) }
        val launcher = rememberFilePickerLauncher(
            type = FileKitType.Image,
            onError = { showError = true },
            onResult = { file ->
                scope.launch {
                    image = file?.toBufferedImage()
                }

            },
        )
        Column(
            modifier = Modifier
                .safeContentPadding()
                .fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(Modifier.height(100.dp))
            Button(onClick = { launcher.launch() }) {
                Text("Pick an image")
            }
        }
    }
}