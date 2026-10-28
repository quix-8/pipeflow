from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .pipeline import OCREngine
from .preprocessing import to_gray
from .structures import Image, RecognizedText

DEFAULT_MNIST_MODEL = os.environ.get("MNIST_MODEL", "/home/quix_/models/mnist-12.onnx")
DEFAULT_TESSDATA = os.environ.get("TESSDATA_PREFIX", "/home/quix_/models/tessdata_best")


class TesseractEngine:
    """Tesseract через tesserocr; текст собирается построчно, confidence — среднее по строкам."""

    def __init__(self, data_path: str = DEFAULT_TESSDATA, language: str = "rus+eng") -> None:
        from tesserocr import PyTessBaseAPI

        self._api = PyTessBaseAPI(path=data_path, lang=language)
        self._lock = threading.Lock()

    def recognize(self, image: Image) -> RecognizedText:
        from PIL import Image as PilImage
        from tesserocr import RIL, iterate_level

        pil = PilImage.fromarray(to_gray(image))
        lines: list[tuple[str, float]] = []
        with self._lock:
            self._api.SetImage(pil)
            self._api.Recognize()
            iterator = self._api.GetIterator()
            if iterator is not None:
                for line in iterate_level(iterator, RIL.TEXTLINE):
                    text = line.GetUTF8Text(RIL.TEXTLINE)
                    if text and text.strip():
                        lines.append((text.strip(), line.Confidence(RIL.TEXTLINE)))
        if not lines:
            return RecognizedText("", 0.0)
        return RecognizedText(
            "\n".join(t for t, _ in lines),
            float(np.mean([c for _, c in lines])) / 100,
        )

    def close(self) -> None:
        self._api.End()


class BlankSkippingEngine:
    """Не вызывает engine на полностью чёрном изображении (пустая ячейка) и возвращает пустой текст."""

    def __init__(self, engine: OCREngine) -> None:
        self.engine = engine

    def recognize(self, image: Image) -> RecognizedText:
        if not image.any():
            return RecognizedText("", 1.0)
        return self.engine.recognize(image)


class OnnxMnistEngine:
    """
    Классификатор одной цифры на модели mnist-12.onnx.
    Ожидает вход после mnist_digit_pre_processor (белая цифра на чёрном); другой размер масштабируется до 28×28.
    """

    SIZE = 28

    def __init__(self, model_path: str = DEFAULT_MNIST_MODEL) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def recognize(self, image: Image) -> RecognizedText:
        gray = to_gray(image)
        if gray.shape != (self.SIZE, self.SIZE):
            gray = cv2.resize(gray, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = (gray.astype(np.float32) / 255.0).reshape(1, 1, self.SIZE, self.SIZE)
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        exp = np.exp(logits - logits.max())
        probabilities = exp / exp.sum()
        digit = int(probabilities.argmax())
        return RecognizedText(str(digit), float(probabilities[digit]))
