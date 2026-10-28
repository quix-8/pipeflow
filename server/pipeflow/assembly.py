from __future__ import annotations

from itertools import groupby
from statistics import fmean

from .structures import Recognition, Rect, TextBlock


def _bounding(items: list[Recognition]) -> Rect:
    left = min(r.region.bounds.x for r in items)
    top = min(r.region.bounds.y for r in items)
    right = max(r.region.bounds.right for r in items)
    bottom = max(r.region.bounds.bottom for r in items)
    return Rect(left, top, right - left, bottom - top)


class LineAssembler:
    """
    Собирает распознанные символы в строки: группирует по вертикальному пересечению,
    внутри строки сортирует слева направо и вставляет пробел на больших разрывах.
    """

    def __init__(self, line_overlap: float = 0.5, space_gap: float = 1.0) -> None:
        self.line_overlap = line_overlap
        self.space_gap = space_gap

    def process(self, recognitions: list[Recognition]) -> list[TextBlock]:
        items = [r for r in recognitions if r.text.strip()]
        lines: list[list[Recognition]] = []
        for item in sorted(items, key=lambda r: r.region.bounds.y):
            line = next((line for line in lines if self._same_line(line, item)), None)
            if line is not None:
                line.append(item)
            else:
                lines.append([item])
        lines.sort(key=lambda line: min(r.region.bounds.y for r in line))
        return [self._to_block(line) for line in lines]

    def _same_line(self, line: list[Recognition], item: Recognition) -> bool:
        top = min(r.region.bounds.y for r in line)
        bottom = max(r.region.bounds.bottom for r in line)
        b = item.region.bounds
        overlap = min(bottom, b.bottom) - max(top, b.y)
        return overlap >= self.line_overlap * min(bottom - top, b.height)

    def _to_block(self, line: list[Recognition]) -> TextBlock:
        ordered = sorted(line, key=lambda r: r.region.bounds.x)
        median_width = sorted(r.region.bounds.width for r in ordered)[len(ordered) // 2]
        text = ""
        for i, rec in enumerate(ordered):
            if i > 0 and rec.region.bounds.x - ordered[i - 1].region.bounds.right > self.space_gap * median_width:
                text += " "
            text += rec.text
        return TextBlock(text, fmean(r.confidence for r in ordered), _bounding(ordered).to_polygon())


class PassThroughAssembler:
    """Каждое непустое распознавание становится отдельным блоком без дополнительной сборки."""

    def process(self, recognitions: list[Recognition]) -> list[TextBlock]:
        items = sorted((r for r in recognitions if r.text.strip()), key=lambda r: r.region.reading_order)
        return [TextBlock(r.text, r.confidence, list(r.region.polygon)) for r in items]


class TableAssembler:
    """Собирает ячейки таблицы в строки: по блоку на строку, ячейки через separator, пустые ячейки сохраняются."""

    def __init__(self, separator: str = " | ") -> None:
        self.separator = separator

    def process(self, recognitions: list[Recognition]) -> list[TextBlock]:
        blocks = []
        by_row = sorted(recognitions, key=lambda r: r.region.row)
        for _, group in groupby(by_row, key=lambda r: r.region.row):
            cells = sorted(group, key=lambda r: r.region.column)
            filled = [c for c in cells if c.text.strip()]
            if not filled:
                continue
            blocks.append(TextBlock(
                self.separator.join(c.text for c in cells),
                fmean(c.confidence for c in filled),
                _bounding(cells).to_polygon(),
            ))
        return blocks
