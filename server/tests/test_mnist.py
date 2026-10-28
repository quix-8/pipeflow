from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import draw_text, font, gradient, to_bgr
from pipeflow.engines import OnnxMnistEngine
from pipeflow.pipelines import mnist_pipeline
from pipeflow.preprocessing import mnist_page_pre_processor
from pipeflow.segmentation import DigitSegmentator
from pipeflow.structures import RecognizedText


class CapturingEngine:
    def __init__(self) -> None:
        self.images: list = []

    def recognize(self, image) -> RecognizedText:
        self.images.append(image)
        return RecognizedText("", 0.0)


def render_digit(digit: str, background=(255, 255, 255), foreground=(0, 0, 0)) -> np.ndarray:
    img = Image.new("RGB", (400, 300), background)
    draw_text(img, digit, 120, 220, 200, foreground)
    return to_bgr(img)


@pytest.mark.parametrize(
    "name,background,foreground,digit",
    [("dark_on_light", (255, 255, 255), (0, 0, 0), "7"), ("light_on_dark", (0, 0, 0), (255, 255, 255), "4")],
)
def test_pre_processor_produces_mnist_input(name, background, foreground, digit, debug_dir):
    engine = CapturingEngine()
    mnist_pipeline(engine).run(render_digit(digit, background, foreground))
    out = engine.images[0]
    cv2.imwrite(str(debug_dir / f"mnist_{name}.png"), out)

    assert out.shape == (28, 28)
    assert out.mean() < 127, "foreground must be the minority (white digit on black)"
    assert out.max() > 200, "digit must be present"
    ys, xs = np.nonzero(out)
    assert 1 <= xs.min() and xs.max() <= 26
    assert 1 <= ys.min() and ys.max() <= 26


def test_recognizes_printed_digits(mnist_model):
    pipeline = mnist_pipeline(OnnxMnistEngine(mnist_model))
    results = [(d, pipeline.run(render_digit(d))) for d in "0123456789"]
    correct = sum([b.text for b in blocks] == [d] for d, blocks in results)
    assert correct >= 7, results


def photo_of_4() -> np.ndarray:
    img = gradient(3000, 2250, (235, 232, 220), (150, 148, 140))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 2999, 119), fill=(60, 60, 60))
    d.rectangle((2900, 0, 2999, 2249), fill=(90, 90, 90))
    color = (40, 40, 70)
    for line in [(1300, 800, 1100, 1350), (1100, 1350, 1600, 1350), (1450, 800, 1450, 1650)]:
        d.line(line, fill=color, width=18)
        for x, y in (line[:2], line[2:]):
            d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
    bgr = to_bgr(img)
    rnd = np.random.default_rng(1)
    xs, ys = rnd.integers(0, 3000, 3000), rnd.integers(0, 2250, 3000)
    bgr[ys, xs] = rnd.integers(60, 120, (3000, 3))
    return bgr


def test_recognizes_photo_like_4(mnist_model):
    blocks = mnist_pipeline(OnnxMnistEngine(mnist_model)).run(photo_of_4())
    assert [b.text for b in blocks] == ["4"]


def photo_of(text: str) -> np.ndarray:
    img = gradient(2400, 900, (235, 232, 220), (160, 158, 150))
    x = 200
    for ch in text:
        draw_text(img, ch, x, 650, 500, (40, 40, 70))
        x += int(font(500).getlength(ch)) + 150
    return to_bgr(img)


def rotated(image: np.ndarray, degrees: float) -> np.ndarray:
    """Поворот по часовой стрелке вокруг центра, как Graphics2D.rotate; при 90° холст меняет ориентацию."""
    h, w = image.shape[:2]
    out_w, out_h = (h, w) if degrees == 90 else (w, h)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    m[0, 2] += (out_w - w) / 2
    m[1, 2] += (out_h - h) / 2
    return cv2.warpAffine(image, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(190, 198, 200))


def test_segments_each_digit():
    prepared = mnist_page_pre_processor().process(photo_of("472"))
    assert len(DigitSegmentator().process(prepared)) == 3


def test_recognizes_several_digits(mnist_model):
    blocks = mnist_pipeline(OnnxMnistEngine(mnist_model)).run(photo_of("472"))
    assert [b.text for b in blocks] == ["472"]


@pytest.mark.parametrize("degrees", [90.0, 8.0])
def test_segments_stacked_and_tilted_digits(degrees):
    prepared = mnist_page_pre_processor().process(rotated(photo_of("4723"), degrees))
    assert len(DigitSegmentator().process(prepared)) == 4
