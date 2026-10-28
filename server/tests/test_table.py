from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import font, gradient, to_bgr
from pipeflow.engines import OnnxMnistEngine
from pipeflow.pipelines import mnist_table_pipeline
from pipeflow.preprocessing import mnist_page_pre_processor
from pipeflow.segmentation import TableSegmentator

ROWS = [["1", "2", "3"], ["4", "", "6"], ["7", "8", "9"], ["0", "3", "5"]]


def photo_of_table(tilt: float) -> np.ndarray:
    w = h = 1800
    background = to_bgr(gradient(w, h, (235, 232, 220), (165, 162, 152)))
    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    left, top, cw, ch = 250, 150, 440, 340
    for r in range(len(ROWS) + 1):
        d.line((left, top + r * ch, left + 3 * cw, top + r * ch), fill=255, width=8)
    for c in range(4):
        d.line((left + c * cw, top, left + c * cw, top + len(ROWS) * ch), fill=255, width=8)
    f = font(220)
    for r, row in enumerate(ROWS):
        for c, text in enumerate(row):
            if text:
                d.text((left + c * cw + cw / 2, top + r * ch + ch / 2), text, font=f, fill=255, anchor="mm")
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -tilt, 1.0)
    ink = cv2.warpAffine(np.asarray(layer), m, (w, h), flags=cv2.INTER_LINEAR)[..., None] / 255.0
    color = np.array((70, 40, 40), np.float32)  # BGR
    return (background * (1 - ink) + color * ink).astype(np.uint8)


@pytest.mark.parametrize("tilt", [0.0, 1.5])
def test_segments_all_cells(tilt):
    regions = TableSegmentator().process(mnist_page_pre_processor().process(photo_of_table(tilt)))
    assert len(regions) == 12
    per_row: dict[int, int] = {}
    for r in regions:
        per_row[r.row] = per_row.get(r.row, 0) + 1
    assert list(per_row.values()) == [3, 3, 3, 3]


@pytest.mark.parametrize("tilt", [0.0, 1.5])
def test_recognizes_table(tilt, mnist_model, debug_dir):
    result = mnist_table_pipeline(OnnxMnistEngine(mnist_model)).run_detailed(photo_of_table(tilt))
    cv2.imwrite(str(debug_dir / f"table_{tilt}.png"), result.prepared)
    assert [b.text for b in result.blocks] == ["1 | 2 | 3", "4 |  | 6", "7 | 8 | 9", "0 | 3 | 5"]
