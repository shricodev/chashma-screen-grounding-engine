import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from chashma.geometry import Point  # noqa: E402
from chashma.model.grounder import Prediction  # noqa: E402
from chashma.serve.app import create_app  # noqa: E402


class FakeGrounder:
    name = "fake"

    def predict(self, image, instruction):
        return Prediction(Point(0.25, 0.5), "<point>250,500</point>", 1.0)


def png(w=200, h=100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_ground_returns_pixels():
    with TestClient(create_app(grounder=FakeGrounder())) as client:
        r = client.post("/v1/ground", files={"image": ("s.png", png())},
                        data={"instruction": "click it"})
    assert r.status_code == 200
    assert (r.json()["x"], r.json()["y"]) == (50, 50)


def test_rejects_bad_input():
    with TestClient(create_app(grounder=FakeGrounder())) as client:
        r = client.post("/v1/ground", files={"image": ("s.png", b"not an image")},
                        data={"instruction": "click"})
        assert r.status_code == 400
        r = client.post("/v1/ground", files={"image": ("s.png", png())},
                        data={"instruction": " "})
        assert r.status_code == 422


def test_http_grounder_roundtrip(monkeypatch):
    import httpx

    from chashma.model.grounder import HTTPGrounder

    with TestClient(create_app(grounder=FakeGrounder())) as client:
        monkeypatch.setattr(httpx, "post", client.post)
        pred = HTTPGrounder("http://testserver").predict(Image.new("RGB", (64, 64)), "click")
    assert pred.point == Point(0.25, 0.5)
