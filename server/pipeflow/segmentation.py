from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import _components, to_gray
from .structures import Image, Rect, Region


@dataclass(frozen=True)
class _Box:
    left: int
    top: int
    right: int
    bottom: int
    area: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def box_area(self) -> int:
        return self.width * self.height

    def intersection_area(self, other: _Box) -> int:
        w = min(self.right, other.right) - max(self.left, other.left)
        h = min(self.bottom, other.bottom) - max(self.top, other.top)
        return w * h if w > 0 and h > 0 else 0

    def union(self, other: _Box) -> _Box:
        return _Box(
            min(self.left, other.left), min(self.top, other.top),
            max(self.right, other.right), max(self.bottom, other.bottom),
            self.area + other.area,
        )


class DigitSegmentator:
    """
    Находит отдельные цифры на бинарном изображении (белые чернила на чёрном) через связные компоненты.
    Части одной цифры склеиваются, только если рамка одной в основном лежит внутри рамки другой;
    это не зависит от ориентации текста.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.0002,
        relative_min_area: float = 0.1,
        merge_overlap: float = 0.5,
        padding: float = 0.1,
    ) -> None:
        self.min_area_ratio = min_area_ratio
        self.relative_min_area = relative_min_area
        self.merge_overlap = merge_overlap
        self.padding = padding

    def process(self, image: Image) -> list[Region]:
        binary = to_gray(image)
        height, width = binary.shape
        _, comps = _components(binary)
        boxes = [_Box(c.x, c.y, c.x + c.w, c.y + c.h, c.area) for c in comps]
        if not boxes:
            return []

        min_area = max(20.0, width * height * self.min_area_ratio)
        sized = [b for b in boxes if b.area >= min_area]
        # компоненты у границы кадра — это края листа и тени, а не цифры
        inner = [b for b in sized if b.left > 0 and b.top > 0 and b.right < width and b.bottom < height]
        pool = inner or sized
        if not pool:
            return []

        largest = max(b.area for b in pool)
        significant = [b for b in pool if b.area >= largest * self.relative_min_area]

        regions = []
        for order, box in enumerate(sorted(self._merge_parts(significant), key=lambda b: b.left)):
            pad = int(max(box.width, box.height) * self.padding)
            x = max(box.left - pad, 0)
            y = max(box.top - pad, 0)
            right = min(box.right + pad, width)
            bottom = min(box.bottom + pad, height)
            regions.append(Region.of(Rect(x, y, right - x, bottom - y), order))
        return regions

    def _merge_parts(self, boxes: list[_Box]) -> list[_Box]:
        result = sorted(boxes, key=lambda b: b.left)
        merged = True
        while merged:
            merged = False
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    a, b = result[i], result[j]
                    if a.intersection_area(b) >= self.merge_overlap * min(a.box_area, b.box_area):
                        result[i] = a.union(b)
                        del result[j]
                        merged = True
                        break
                if merged:
                    break
        return result


class WholeImageSegmentator:
    """Один регион на всё изображение: для движков, которые сами разбирают страницу (Tesseract)."""

    def process(self, image: Image) -> list[Region]:
        h, w = image.shape[:2]
        return [Region.of(Rect(0, 0, w, h))]


@dataclass(frozen=True)
class _Cell:
    x: int
    y: int
    w: int
    h: int

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2


class TableSegmentator:
    """
    Находит ячейки таблицы с видимой сеткой на бинарном изображении (белые линии/чернила на чёрном).
    Линии выделяются морфологическим открытием, остаются только длинные (сопоставимые с самой длинной),
    ячейки — замкнутые области между линиями. Каждая ячейка отдаётся с отступом от линий,
    row/column задают позицию, reading_order — порядок по строкам.
    Таблица должна быть замкнута линиями и не сильно наклонена (до пары градусов).
    """

    def __init__(
        self,
        line_kernel_ratio: float = 1 / 30,
        min_line_ratio: float = 0.5,
        min_cell_area_ratio: float = 0.0005,
        inset: float = 0.1,
    ) -> None:
        self.line_kernel_ratio = line_kernel_ratio
        self.min_line_ratio = min_line_ratio
        self.min_cell_area_ratio = min_cell_area_ratio
        self.inset = inset

    def process(self, image: Image) -> list[Region]:
        gray = to_gray(image)
        height, width = gray.shape
        grid = cv2.bitwise_or(self._lines(gray, horizontal=True), self._lines(gray, horizontal=False))
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        grid = cv2.bitwise_not(grid)
        return self._to_regions(self._cells(grid, width, height))

    def _lines(self, image: Image, horizontal: bool) -> Image:
        rows, cols = image.shape
        length = max(int((cols if horizontal else rows) * self.line_kernel_ratio), 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        opened = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        labels, comps = _components(opened)
        extent = {c.index: c.w if horizontal else c.h for c in comps}
        longest = max(extent.values(), default=0)
        kept = [i for i, e in extent.items() if e >= longest * self.min_line_ratio]
        return np.isin(labels, kept).astype(np.uint8) * 255

    def _cells(self, grid: Image, width: int, height: int) -> list[_Cell]:
        _, comps = _components(grid, connectivity=4)
        min_area = width * height * self.min_cell_area_ratio
        cells = []
        for c in comps:
            touches_border = c.x == 0 or c.y == 0 or c.x + c.w >= width or c.y + c.h >= height
            solid = c.area >= 0.5 * c.w * c.h
            if not touches_border and solid and c.area >= min_area:
                cells.append(_Cell(c.x, c.y, c.w, c.h))
        return cells

    def _to_regions(self, cells: list[_Cell]) -> list[Region]:
        rows: list[list[_Cell]] = []
        for cell in sorted(cells, key=lambda c: c.center_y):
            row = rows[-1] if rows else None
            if row is not None and abs(cell.center_y - sum(c.center_y for c in row) / len(row)) < 0.5 * cell.h:
                row.append(cell)
            else:
                rows.append([cell])
        regions = []
        order = 0
        for r, row in enumerate(rows):
            for c, cell in enumerate(sorted(row, key=lambda c: c.x)):
                dx = int(cell.w * self.inset)
                dy = int(cell.h * self.inset)
                regions.append(Region.of(Rect(cell.x + dx, cell.y + dy, cell.w - 2 * dx, cell.h - 2 * dy), order, r, c))
                order += 1
        return regions
