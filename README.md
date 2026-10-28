# Pipeflow

Распознавание цифр, таблиц и текста. Проект состоит из двух частей:

- [`server/`](./server) — Python-сервер (FastAPI) со всей ML-частью: предобработка OpenCV, сегментация,
  MNIST через onnxruntime, Tesseract через tesserocr, сборка результата.
- [`shared/`](./shared/src) + [`desktopApp/`](./desktopApp) — Compose Desktop клиент на Kotlin, который отправляет
  изображение на сервер (Ktor) и показывает результат.

## Сервер

```bash
cd server
uv sync --extra dev          # или: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
uv run pipeflow-server       # http://127.0.0.1:8000, документация API: /docs
uv run pytest
```

Переменные окружения: `MNIST_MODEL` (путь к `mnist-12.onnx`), `TESSDATA_PREFIX` (каталог с `rus`/`eng.traineddata`),
`PIPEFLOW_HOST`, `PIPEFLOW_PORT`.

API:
- `POST /recognize?pipeline=mnist|mnist_table|tesseract` — multipart-поле `file` с изображением. Ответ содержит
  подготовленное изображение (`preparedPng`, base64), распознанные регионы и собранные блоки текста.
- `GET /pipelines`, `GET /health`.

## Клиент

Приложение само запускает сервер (`LocalServer`): ищет каталог `server/`, при первом запуске создаёт `.venv`
и ставит зависимости (через `uv`, если он есть, иначе через `python -m venv` + `pip`), выбирает свободный порт
и ждёт готовности, показывая статус в окне. Логи пишутся в `server/server.log` и `server/install.log`.
Сервер завершается вместе с приложением, в том числе при аварийном закрытии (он следит за stdin).

- `PIPEFLOW_SERVER` — использовать уже запущенный сервер и ничего не стартовать.
- `PIPEFLOW_SERVER_DIR` — путь к каталогу `server/`, если он не лежит рядом с проектом.

This is a Kotlin Multiplatform project targeting Desktop (JVM).

* [/shared](./shared/src) is for code that will be shared across your Compose Multiplatform applications. It contains
  several subfolders:
    - [commonMain](./shared/src/commonMain/kotlin) is for code that’s common for all targets.
    - Other folders are for Kotlin code that will be compiled for only the platform indicated in the folder name. For
      example, if you want to use Apple’s CoreCrypto for the iOS part of your Kotlin app,
      the [iosMain](./shared/src/iosMain/kotlin) folder would be the right place for such calls. Similarly, if you want
      to edit the Desktop (JVM) specific part, the [jvmMain](./shared/src/jvmMain/kotlin)
      folder is the appropriate location.

### Running the apps

Use the run configurations provided by the run widget in your IDE's toolbar. You can also use these commands and
options:

- Desktop app:
    - Hot reload: `./gradlew :desktopApp:hotRun --auto`
    - Standard run: `./gradlew :desktopApp:run`

---

Learn more about [Kotlin Multiplatform](https://www.jetbrains.com/help/kotlin-multiplatform-dev/get-started.html)…