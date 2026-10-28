package com.quix.pipeflow.modules

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.plugins.timeout
import io.ktor.client.request.forms.formData
import io.ktor.client.request.forms.submitFormWithBinaryData
import io.ktor.client.request.get
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.Headers
import io.ktor.http.HttpHeaders
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class OcrServerException(message: String) : Exception(message)

/** HTTP-клиент Python-сервера распознавания (см. каталог server/). */
class OcrClient(baseUrl: String) : AutoCloseable {
    private val json = Json { ignoreUnknownKeys = true }

    private val http = HttpClient(CIO) {
        install(ContentNegotiation) { json(json) }
        install(HttpTimeout) { requestTimeoutMillis = 120_000 }
        defaultRequest { url(baseUrl.trimEnd('/') + "/") }
    }

    suspend fun recognize(image: ByteArray, fileName: String, pipeline: PipelineKind): RecognizeResponse {
        val response = http.submitFormWithBinaryData(
            url = "recognize",
            formData = formData {
                append(
                    "file", image,
                    Headers.build {
                        append(HttpHeaders.ContentType, ContentType.Application.OctetStream.toString())
                        append(HttpHeaders.ContentDisposition, "filename=\"$fileName\"")
                    },
                )
            },
        ) {
            url { parameters.append("pipeline", pipeline.serialName) }
        }
        if (!response.status.isSuccess()) throw OcrServerException(errorMessage(response.bodyAsText(), response.status.value))
        return response.body()
    }

    suspend fun isAvailable(): Boolean = runCatching {
        http.get("health") { timeout { requestTimeoutMillis = 2_000 } }.status.isSuccess()
    }.getOrDefault(false)

    override fun close() = http.close()

    private fun errorMessage(body: String, status: Int): String {
        val detail = runCatching { json.parseToJsonElement(body).jsonObject["detail"]?.jsonPrimitive?.content }.getOrNull()
        return "HTTP $status: ${detail ?: body}"
    }

    private val PipelineKind.serialName: String
        get() = when (this) {
            PipelineKind.MNIST -> "mnist"
            PipelineKind.MNIST_TABLE -> "mnist_table"
            PipelineKind.TESSERACT -> "tesseract"
        }
}
