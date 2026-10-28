from __future__ import annotations

from .assembly import LineAssembler, PassThroughAssembler, TableAssembler
from .engines import BlankSkippingEngine
from .pipeline import OCREngine, Pipeline, Recognizer
from .preprocessing import (
    mnist_cell_pre_processor,
    mnist_digit_pre_processor,
    mnist_page_pre_processor,
    tesseract_pre_processor,
)
from .segmentation import DigitSegmentator, TableSegmentator, WholeImageSegmentator


def mnist_pipeline(engine: OCREngine) -> Pipeline:
    return Pipeline(
        mnist_page_pre_processor(),
        DigitSegmentator(),
        Recognizer(engine, mnist_digit_pre_processor()),
        LineAssembler(),
    )


def mnist_table_pipeline(engine: OCREngine) -> Pipeline:
    return Pipeline(
        mnist_page_pre_processor(),
        TableSegmentator(),
        Recognizer(BlankSkippingEngine(engine), mnist_cell_pre_processor()),
        TableAssembler(),
    )


def tesseract_pipeline(engine: OCREngine) -> Pipeline:
    return Pipeline(
        tesseract_pre_processor(),
        WholeImageSegmentator(),
        Recognizer(engine),
        PassThroughAssembler(),
    )
