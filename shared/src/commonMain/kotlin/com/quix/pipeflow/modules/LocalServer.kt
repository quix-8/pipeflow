package com.quix.pipeflow.modules

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import java.io.File
import java.net.ServerSocket
import kotlin.coroutines.coroutineContext
import kotlin.time.Duration.Companion.seconds
import kotlin.time.TimeSource

sealed interface ServerState {
    data class Starting(val message: String) : ServerState
    data class Ready(val client: OcrClient) : ServerState
    data class Failed(val message: String) : ServerState
}

/**
 * Поднимает Python-сервер распознавания как дочерний процесс: при первом запуске создаёт окружение,
 * выбирает свободный порт, ждёт готовности и завершает процесс при закрытии.
 * Если задана переменная PIPEFLOW_SERVER, используется внешний сервер и ничего не запускается.
 */
class LocalServer : AutoCloseable {
    private val _state = MutableStateFlow<ServerState>(ServerState.Starting("Запуск сервера…"))
    val state: StateFlow<ServerState> = _state.asStateFlow()

    private var process: Process? = null
    private val shutdownHook = Thread { process?.destroyForcibly() }

    suspend fun start() {
        _state.value = ServerState.Starting("Запуск сервера…")
        process?.destroyForcibly()
        process = null
        try {
            val external = System.getenv("PIPEFLOW_SERVER")
            if (external != null) {
                awaitReady(OcrClient(external), null)
                return
            }
            val dir = findServerDir()
                ?: throw IllegalStateException("Не найден каталог server/ (задайте PIPEFLOW_SERVER_DIR)")
            val python = withContext(Dispatchers.IO) { ensureEnvironment(dir) }
            val port = ServerSocket(0).use { it.localPort }
            val log = File(dir, "server.log")
            _state.value = ServerState.Starting("Запуск сервера…")
            val started = withContext(Dispatchers.IO) {
                ProcessBuilder(python, "-m", "pipeflow.api")
                    .directory(dir)
                    .redirectErrorStream(true)
                    .redirectOutput(log)
                    .apply {
                        environment()["PIPEFLOW_HOST"] = "127.0.0.1"
                        environment()["PIPEFLOW_PORT"] = port.toString()
                        environment()["PIPEFLOW_WATCH_STDIN"] = "1"
                    }
                    .start()
            }
            process = started
            Runtime.getRuntime().addShutdownHook(shutdownHook)
            awaitReady(OcrClient("http://127.0.0.1:$port"), started, log)
        } catch (e: Exception) {
            _state.value = ServerState.Failed(e.message ?: e::class.simpleName.orEmpty())
        }
    }

    private suspend fun awaitReady(client: OcrClient, process: Process?, log: File? = null) {
        val deadline = TimeSource.Monotonic.markNow() + 120.seconds
        while (coroutineContext.isActive) {
            if (client.isAvailable()) {
                _state.value = ServerState.Ready(client)
                return
            }
            if (process != null && !process.isAlive) {
                client.close()
                throw IllegalStateException("Сервер завершился (код ${process.exitValue()}):\n${log?.tail().orEmpty()}")
            }
            if (deadline.hasPassedNow()) {
                client.close()
                throw IllegalStateException("Сервер не ответил за отведённое время")
            }
            delay(300)
        }
    }

    /** Возвращает путь к python из server/.venv, при необходимости создавая окружение. */
    private fun ensureEnvironment(dir: File): String {
        val windows = System.getProperty("os.name").lowercase().contains("win")
        val python = File(dir, if (windows) ".venv/Scripts/python.exe" else ".venv/bin/python")
        val marker = File(dir, ".venv/.pipeflow-installed")
        val pyproject = File(dir, "pyproject.toml")
        if (python.exists() && marker.exists() && marker.lastModified() >= pyproject.lastModified()) {
            return python.path
        }

        _state.value = ServerState.Starting("Первый запуск: установка зависимостей сервера…")
        val log = File(dir, "install.log")
        fun run(vararg command: String) {
            val p = ProcessBuilder(*command).directory(dir).redirectErrorStream(true).redirectOutput(log).start()
            if (p.waitFor() != 0) {
                throw IllegalStateException("Не удалось установить зависимости (${command.joinToString(" ")}):\n${log.tail()}")
            }
        }
        val uv = findExecutable("uv")
        if (uv != null) {
            run(uv, "sync")
        } else {
            if (!python.exists()) {
                val system = findExecutable("python3") ?: findExecutable("python")
                    ?: throw IllegalStateException("Не найден Python 3.11+ или uv")
                run(system, "-m", "venv", ".venv")
            }
            run(python.path, "-m", "pip", "install", "-e", ".")
        }
        marker.writeText("")
        return python.path
    }

    private fun findServerDir(): File? {
        val candidates = listOfNotNull(
            System.getenv("PIPEFLOW_SERVER_DIR"),
            System.getProperty("pipeflow.serverDir"),
        ).map(::File) + generateSequence(File("").absoluteFile) { it.parentFile }.map { File(it, "server") }
        return candidates.firstOrNull { File(it, "pyproject.toml").exists() }
    }

    private fun findExecutable(name: String): String? {
        val names = listOf(name, "$name.exe")
        return System.getenv("PATH").orEmpty().split(File.pathSeparator)
            .flatMap { dir -> names.map { File(dir, it) } }
            .firstOrNull { it.isFile && it.canExecute() }?.path
    }

    private fun File.tail(lines: Int = 15): String =
        runCatching { readLines().takeLast(lines).joinToString("\n") }.getOrDefault("")

    override fun close() {
        (_state.value as? ServerState.Ready)?.client?.close()
        process?.let {
            it.destroy()
            if (!it.waitFor(3, java.util.concurrent.TimeUnit.SECONDS)) it.destroyForcibly()
            runCatching { Runtime.getRuntime().removeShutdownHook(shutdownHook) }
        }
        process = null
    }
}
