from __future__ import annotations

import io
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from chashma import __version__
from chashma.model.grounder import Grounder, load_grounder

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
MAX_INSTRUCTION_CHARS = 500


class GroundResponse(BaseModel):
    x: int
    y: int
    x_norm: float
    y_norm: float
    width: int
    height: int
    model: str
    latency_ms: float
    raw: str


def create_app(model: str | None = None, grounder: Grounder | None = None) -> FastAPI:
    state: dict[str, Grounder] = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state["grounder"] = grounder or load_grounder(
            model or os.environ.get("CHASHMA_MODEL", "checkpoints/chashma-256m")
        )
        yield
        state.clear()

    app = FastAPI(title="chashma", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        g = state.get("grounder")
        return {"status": "ok" if g else "loading", "model": g.name if g else None}

    @app.post("/v1/ground", response_model=GroundResponse)
    async def ground(
        image: UploadFile = File(...), instruction: str = Form(...)
    ) -> GroundResponse:
        instruction = instruction.strip()
        if not instruction or len(instruction) > MAX_INSTRUCTION_CHARS:
            raise HTTPException(422, f"instruction must be 1..{MAX_INSTRUCTION_CHARS} chars")
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image too large")
        try:
            img = Image.open(io.BytesIO(data))
            if img.width * img.height > MAX_PIXELS:
                raise HTTPException(413, "image has too many pixels")
            img = img.convert("RGB")
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
            raise HTTPException(400, "could not decode image") from e

        g = state["grounder"]

        def run():
            with lock:
                return g.predict(img, instruction)

        pred = await run_in_threadpool(run)
        if pred.point is None:
            raise HTTPException(502, f"model returned no point: {pred.raw[:200]!r}")
        x, y = pred.point.to_pixels(img.width, img.height)
        return GroundResponse(
            x=x,
            y=y,
            x_norm=round(pred.point.x, 5),
            y_norm=round(pred.point.y, 5),
            width=img.width,
            height=img.height,
            model=g.name,
            latency_ms=round(pred.latency_ms, 1),
            raw=pred.raw,
        )

    return app
