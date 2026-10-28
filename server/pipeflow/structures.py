from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Image = np.ndarray
"""Изображение OpenCV: uint8, 2D (градации серого) или 3D BGR."""


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def to_polygon(self) -> list[Point]:
        return [
            Point(self.x, self.y),
            Point(self.right, self.y),
            Point(self.right, self.bottom),
            Point(self.x, self.bottom),
        ]


@dataclass(frozen=True)
class Region:
    """Область изображения, найденная сегментатором. polygon допускает повёрнутый текст."""

    polygon: list[Point]
    reading_order: int = 0
    row: int = 0
    column: int = 0

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError("Region polygon must have at least 3 points")

    @property
    def bounds(self) -> Rect:
        xs = [p.x for p in self.polygon]
        ys = [p.y for p in self.polygon]
        return Rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    @staticmethod
    def of(rect: Rect, reading_order: int = 0, row: int = 0, column: int = 0) -> Region:
        return Region(rect.to_polygon(), reading_order, row, column)


@dataclass(frozen=True)
class RecognizedText:
    """Результат работы движка над одним изображением; confidence в диапазоне 0..1."""

    text: str
    confidence: float


@dataclass(frozen=True)
class Recognition:
    """Результат распознавания одного региона."""

    region: Region
    text: str
    confidence: float


@dataclass(frozen=True)
class TextBlock:
    """Итоговый текстовый блок после сборки (строка/абзац)."""

    text: str
    confidence: float
    bounding_box: list[Point] = field(default_factory=list)
