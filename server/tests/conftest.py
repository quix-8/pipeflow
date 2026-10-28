from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from pipeflow.engines import DEFAULT_MNIST_MODEL, DEFAULT_TESSDATA

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def font(size: int, path: str | None = None) -> ImageFont.FreeTypeFont:
    for candidate in [path, os.environ.get("PIPEFLOW_TEST_FONT"), *FONT_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    pytest.skip("no TrueType font found")


def to_bgr(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def gradient(w: int, h: int, start: tuple[int, int, int], end: tuple[int, int, int]) -> Image.Image:
    """Диагональный градиент как java.awt.GradientPaint от (0,0) до (w,h)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.clip((xx * w + yy * h) / (w * w + h * h), 0, 1)[..., None]
    rgb = np.array(start, np.float32) * (1 - t) + np.array(end, np.float32) * t
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def draw_text(img: Image.Image, text: str, x: int, baseline: int, size: int, fill) -> None:
    ImageDraw.Draw(img).text((x, baseline), text, font=font(size), fill=fill, anchor="ls")


@pytest.fixture
def mnist_model() -> str:
    if not Path(DEFAULT_MNIST_MODEL).exists():
        pytest.skip(f"MNIST model not found: {DEFAULT_MNIST_MODEL}")
    return DEFAULT_MNIST_MODEL


@pytest.fixture
def tessdata() -> str:
    if not Path(DEFAULT_TESSDATA).exists():
        pytest.skip(f"tessdata not found: {DEFAULT_TESSDATA}")
    return DEFAULT_TESSDATA


@pytest.fixture
def debug_dir() -> Path:
    path = Path("/tmp/opencode")
    path.mkdir(parents=True, exist_ok=True)
    return path
