from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from pipeflow.api import app, decode_image
from test_mnist import photo_of

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_pipelines():
    assert client.get("/pipelines").json() == ["mnist", "mnist_table", "tesseract"]


def test_rejects_garbage():
    response = client.post("/recognize", files={"file": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_decode_composites_alpha_on_white():
    rgba = np.zeros((4, 4, 4), np.uint8)
    _, png = cv2.imencode(".png", rgba)
    assert (decode_image(png.tobytes()) == 255).all()


def test_recognize_mnist(mnist_model):
    _, png = cv2.imencode(".png", photo_of("472"))
    response = client.post("/recognize?pipeline=mnist", files={"file": ("digits.png", png.tobytes(), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert [b["text"] for b in body["blocks"]] == ["472"]
    assert len(body["recognitions"]) == 3
    prepared = cv2.imdecode(np.frombuffer(base64.b64decode(body["preparedPng"]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert prepared.shape[:2] == (body["height"], body["width"])
