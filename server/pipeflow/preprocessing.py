from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import PreProcessor, PreProcessorChain
from .structures import Image


def to_gray(image: Image) -> Image:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@dataclass
class _Component:
    index: int
    x: int
    y: int
    w: int
    h: int
    area: int


def _components(binary: Image, connectivity: int = 8) -> tuple[Image, list[_Component]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity, ltype=cv2.CV_32S)
    comps = [
        _Component(
            i,
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        for i in range(1, count)
    ]
    return labels, comps


class CenterDigitStep:
    """Вписывает цифру в квадрат box×box и центрирует по центру масс на холсте size×size, как в MNIST."""

    def __init__(self, size: int = 28, box: int = 20) -> None:
        self.size = size
        self.box = box

    def process(self, image: Image) -> Image:
        gray = to_gray(image)
        canvas = np.zeros((self.size, self.size), np.uint8)
        points = cv2.findNonZero(gray)
        if points is None:
            return canvas
        x, y, bw, bh = cv2.boundingRect(points)
        scale = self.box / max(bw, bh)
        w = max(int(bw * scale), 1)
        h = max(int(bh * scale), 1)
        resized = cv2.resize(gray[y:y + bh, x:x + bw], (w, h), interpolation=cv2.INTER_AREA)
        left = (self.size - w) // 2
        top = (self.size - h) // 2
        canvas[top:top + h, left:left + w] = resized
        return self._shift_to_center_of_mass(canvas)

    @staticmethod
    def _shift_to_center_of_mass(image: Image) -> Image:
        m = cv2.moments(image, binaryImage=False)
        if m["m00"] == 0:
            return image
        rows, cols = image.shape
        dx = math.floor(cols / 2 - m["m10"] / m["m00"] + 0.5)
        dy = math.floor(rows / 2 - m["m01"] / m["m00"] + 0.5)
        shift = np.array([[1, 0, dx], [0, 1, dy]], np.float64)
        return cv2.warpAffine(image, shift, (cols, rows))


class ResizeStep:
    """Уменьшает изображение так, чтобы большая сторона не превышала max_side."""

    def __init__(self, max_side: int = 1000) -> None:
        self.max_side = max_side

    def process(self, image: Image) -> Image:
        h, w = image.shape[:2]
        side = max(w, h)
        if side <= self.max_side:
            return image
        scale = self.max_side / side
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class AdaptiveBinarizeStep:
    """Адаптивная бинаризация для фотографий с размытием и удалением неоднородностей фона."""

    def __init__(self, window_ratio: float = 0.25, offset: float = 10.0) -> None:
        self.window_ratio = window_ratio
        self.offset = offset

    def process(self, image: Image) -> Image:
        gray = to_gray(image).copy()
        if gray.mean() < 100:
            gray = cv2.bitwise_not(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        window = max(int(min(gray.shape) * self.window_ratio), 15) | 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, window, self.offset,
        )


class ExtractDigitStep:
    """Оставляет самую крупную компоненту и её части рядом, вырезает их в квадрат с отступом."""

    def __init__(self, min_area_ratio: float = 0.0002, part_area_ratio: float = 0.1, padding: float = 0.1) -> None:
        self.min_area_ratio = min_area_ratio
        self.part_area_ratio = part_area_ratio
        self.padding = padding

    def process(self, image: Image) -> Image:
        binary = to_gray(image)
        rows, cols = binary.shape
        labels, comps = _components(binary)
        min_area = max(20.0, rows * cols * self.min_area_ratio)
        sized = [c for c in comps if c.area >= min_area]
        # компоненты у границы кадра — это края листа и тени, а не цифра
        inner = [c for c in sized if c.x > 0 and c.y > 0 and c.x + c.w < cols and c.y + c.h < rows]
        pool = inner or sized
        if not pool:
            return binary.copy()

        main = max(pool, key=lambda c: c.area)
        margin = max(main.w, main.h) // 2
        kept = [
            c for c in pool
            if c is main or (
                c.area >= main.area * self.part_area_ratio
                and main.x - margin <= c.x + c.w // 2 <= main.x + main.w + margin
                and main.y - margin <= c.y + c.h // 2 <= main.y + main.h + margin
            )
        ]
        mask = np.isin(labels, [c.index for c in kept]).astype(np.uint8) * 255

        left = min(c.x for c in kept)
        top = min(c.y for c in kept)
        w = max(c.x + c.w for c in kept) - left
        h = max(c.y + c.h for c in kept) - top
        side = int(max(w, h) * (1 + 2 * self.padding))
        square = np.zeros((side, side), np.uint8)
        ox = (side - w) // 2
        oy = (side - h) // 2
        square[oy:oy + h, ox:ox + w] = mask[top:top + h, left:left + w]
        return square


class ThickenStep:
    """Утолщение штрихов цифры для фотографий."""

    def __init__(self, ratio: float = 0.06) -> None:
        self.ratio = ratio

    def process(self, image: Image) -> Image:
        gray = to_gray(image)
        k = max(int(max(gray.shape) * self.ratio), 1) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(gray, kernel)


class TrimBordersStep:
    """Обрезка пустых полей вокруг всего «чернильного» содержимого; цвета не трогает."""

    def __init__(self, padding: float = 0.02) -> None:
        self.padding = padding

    def process(self, image: Image) -> Image:
        gray = to_gray(image)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if binary.mean() > 127:
            binary = cv2.bitwise_not(binary)
        points = cv2.findNonZero(binary)
        if points is None:
            return gray.copy()
        x, y, bw, bh = cv2.boundingRect(points)
        pad = max(int(max(bw, bh) * self.padding), 1)
        rows, cols = gray.shape
        left, top = max(x - pad, 0), max(y - pad, 0)
        right, bottom = min(x + bw + pad, cols), min(y + bh + pad, rows)
        return gray[top:bottom, left:right].copy()


def mnist_page_pre_processor() -> PreProcessor:
    """Предобработка всего кадра для многоцифровой фотографии: на выходе белые цифры на чёрном."""
    return PreProcessorChain(ResizeStep(), AdaptiveBinarizeStep())


def mnist_digit_pre_processor() -> PreProcessor:
    """Предобработка вырезанной цифры перед подачей в MNIST."""
    return PreProcessorChain(ThickenStep(), CenterDigitStep())


def mnist_cell_pre_processor() -> PreProcessor:
    """Предобработка вырезанной ячейки таблицы с одной цифрой. Пустая ячейка превращается в чёрный кадр."""
    return PreProcessorChain(ExtractDigitStep(), ThickenStep(), CenterDigitStep())


def tesseract_pre_processor() -> PreProcessor:
    """Для Tesseract: он сам работает с градациями серого, поэтому только масштаб и обрезка полей."""
    return PreProcessorChain(ResizeStep(max_side=2500), TrimBordersStep())
