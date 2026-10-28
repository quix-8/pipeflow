from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import font, to_bgr
from pipeflow.engines import TesseractEngine
from pipeflow.pipeline import Recognizer
from pipeflow.pipelines import tesseract_pipeline
from pipeflow.structures import Rect, Region

TEXT = "Hello Привет 12345"


def render(text: str = TEXT) -> np.ndarray:
    img = Image.new("RGB", (900, 160), (255, 255, 255))
    ImageDraw.Draw(img).text((30, 100), text, font=font(64, "/usr/share/fonts/truetype/NotoSans-Regular.ttf"), fill=(0, 0, 0), anchor="ls")
    return to_bgr(img)


@pytest.fixture(scope="module")
def engine():
    from pipeflow.engines import DEFAULT_TESSDATA
    from pathlib import Path

    if not Path(DEFAULT_TESSDATA).exists():
        pytest.skip("tessdata not found")
    e = TesseractEngine()
    yield e
    e.close()


def test_recognizes_english_and_russian(engine):
    text = engine.recognize(render()).text
    for word in ["Hello", "Привет", "12345"]:
        assert word in text, text


def test_recognizer_crops_regions(engine):
    image = render()
    result = Recognizer(engine).process(image, [Region.of(Rect(0, 0, image.shape[1], image.shape[0]))])
    assert "Hello" in result[0].text


def test_pipeline_trims_margins_and_recognizes(engine):
    image = np.full((700, 1600, 3), 255, np.uint8)
    image[300:460, 400:1300] = render()
    result = tesseract_pipeline(engine).run_detailed(image)
    assert result.prepared.shape[1] < 1600 and result.prepared.shape[0] < 700
    assert len(result.blocks) == 1 and "Hello" in result.blocks[0].text
