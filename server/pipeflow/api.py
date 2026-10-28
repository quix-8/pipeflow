from __future__ import annotations

import base64
import time
from enum import Enum
from functools import cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .engines import OnnxMnistEngine, TesseractEngine
from .pipeline import Pipeline, PipelineResult
from .pipelines import mnist_pipeline, mnist_table_pipeline, tesseract_pipeline
from .structures import Image, Point


class PipelineKind(str, Enum):
    MNIST = "mnist"
    MNIST_TABLE = "mnist_table"
    TESSERACT = "tesseract"


class PointDto(BaseModel):
    x: int
    y: int


class RegionDto(BaseModel):
    polygon: list[PointDto]
    readingOrder: int
    row: int
    column: int


class RecognitionDto(BaseModel):
    region: RegionDto
    text: str
    confidence: float


class TextBlockDto(BaseModel):
    text: str
    confidence: float
    boundingBox: list[PointDto]


class RecognizeResponse(BaseModel):
    pipeline: PipelineKind
    preparedPng: str
    """Подготовленное изображение (после предобработки) в PNG, base64; координаты регионов относятся к нему."""
    width: int
    height: int
    recognitions: list[RecognitionDto]
    blocks: list[TextBlockDto]
    processingTimeMs: int


@cache
def _mnist_engine() -> OnnxMnistEngine:
    return OnnxMnistEngine()


@cache
def get_pipeline(kind: PipelineKind) -> Pipeline:
    match kind:
        case PipelineKind.MNIST:
            return mnist_pipeline(_mnist_engine())
        case PipelineKind.MNIST_TABLE:
            return mnist_table_pipeline(_mnist_engine())
        case PipelineKind.TESSERACT:
            return tesseract_pipeline(TesseractEngine())


def decode_image(data: bytes) -> Image:
    """Декодирует файл в BGR; прозрачность накладывается на белый фон."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Не удалось декодировать изображение")
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image, alpha=255.0 / np.iinfo(image.dtype).max)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        rgb = image[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return image


def _points(points: list[Point]) -> list[PointDto]:
    return [PointDto(x=p.x, y=p.y) for p in points]


def to_response(kind: PipelineKind, result: PipelineResult, elapsed_ms: int) -> RecognizeResponse:
    ok, png = cv2.imencode(".png", result.prepared)
    if not ok:
        raise RuntimeError("Не удалось закодировать изображение")
    h, w = result.prepared.shape[:2]
    return RecognizeResponse(
        pipeline=kind,
        preparedPng=base64.b64encode(png.tobytes()).decode(),
        width=w,
        height=h,
        recognitions=[
            RecognitionDto(
                region=RegionDto(
                    polygon=_points(r.region.polygon),
                    readingOrder=r.region.reading_order,
                    row=r.region.row,
                    column=r.region.column,
                ),
                text=r.text,
                confidence=r.confidence,
            )
            for r in result.recognitions
        ],
        blocks=[TextBlockDto(text=b.text, confidence=b.confidence, boundingBox=_points(b.bounding_box)) for b in result.blocks],
        processingTimeMs=elapsed_ms,
    )


app = FastAPI(title="Pipeflow OCR")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/pipelines")
def pipelines() -> list[PipelineKind]:
    return list(PipelineKind)


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(
    file: UploadFile = File(...),
    pipeline: PipelineKind = Query(PipelineKind.MNIST),
) -> RecognizeResponse:
    data = await file.read()
    try:
        image = decode_image(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def run() -> RecognizeResponse:
        started = time.perf_counter()
        result = get_pipeline(pipeline).run_detailed(image)
        return to_response(pipeline, result, int((time.perf_counter() - started) * 1000))

    try:
        return await run_in_threadpool(run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e


def _exit_when_parent_closes_stdin() -> None:
    """Клиент держит stdin открытым; когда он закрылся (клиент завершился, в том числе аварийно), сервер выходит."""
    import os
    import sys
    import threading

    def watch() -> None:
        try:
            sys.stdin.buffer.read()
        finally:
            os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    import os

    import uvicorn

    if os.environ.get("PIPEFLOW_WATCH_STDIN"):
        _exit_when_parent_closes_stdin()
    uvicorn.run(
        app,
        host=os.environ.get("PIPEFLOW_HOST", "127.0.0.1"),
        port=int(os.environ.get("PIPEFLOW_PORT", "8000")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
