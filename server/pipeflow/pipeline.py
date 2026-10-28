from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .structures import Image, RecognizedText, Recognition, Region, TextBlock


class PreProcessor(Protocol):
    def process(self, image: Image) -> Image: ...


class Segmentator(Protocol):
    def process(self, image: Image) -> list[Region]: ...


class Assembler(Protocol):
    def process(self, recognitions: list[Recognition]) -> list[TextBlock]: ...


class OCREngine(Protocol):
    def recognize(self, image: Image) -> RecognizedText: ...


class PreProcessorChain:
    """Последовательно применяет шаги предобработки; пустая цепочка возвращает изображение без изменений."""

    def __init__(self, *steps: PreProcessor) -> None:
        self.steps = list(steps)

    def process(self, image: Image) -> Image:
        for step in self.steps:
            image = step.process(image)
        return image


class Recognizer:
    def __init__(self, engine: OCREngine, crop_pre_processor: PreProcessor | None = None) -> None:
        self.engine = engine
        self.crop_pre_processor = crop_pre_processor or PreProcessorChain()

    def process(self, image: Image, regions: list[Region]) -> list[Recognition]:
        result = []
        for region in regions:
            crop = _crop(image, region)
            if crop is None:
                rec = RecognizedText("", 0.0)
            else:
                rec = self.engine.recognize(self.crop_pre_processor.process(crop))
            result.append(Recognition(region, rec.text, rec.confidence))
        return result


def _crop(image: Image, region: Region) -> Image | None:
    h, w = image.shape[:2]
    b = region.bounds
    x = min(max(b.x, 0), w)
    y = min(max(b.y, 0), h)
    right = min(max(b.right, 0), w)
    bottom = min(max(b.bottom, 0), h)
    if right <= x or bottom <= y:
        return None
    return image[y:bottom, x:right].copy()


@dataclass
class PipelineResult:
    """Полный результат прогона: подготовленное изображение, найденные регионы и собранные блоки."""

    prepared: Image
    recognitions: list[Recognition]
    blocks: list[TextBlock]


class Pipeline:
    def __init__(
        self,
        pre_processor: PreProcessor,
        segmentator: Segmentator,
        recognizer: Recognizer,
        assembler: Assembler,
    ) -> None:
        self.pre_processor = pre_processor
        self.segmentator = segmentator
        self.recognizer = recognizer
        self.assembler = assembler

    def run(self, image: Image) -> list[TextBlock]:
        return self.run_detailed(image).blocks

    def run_detailed(self, image: Image) -> PipelineResult:
        prepared = self.pre_processor.process(image)
        regions = self.segmentator.process(prepared)
        recognitions = self.recognizer.process(prepared, regions)
        return PipelineResult(prepared, recognitions, self.assembler.process(recognitions))
