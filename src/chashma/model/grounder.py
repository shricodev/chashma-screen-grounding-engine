from __future__ import annotations

import asyncio
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from chashma.geometry import Point
from chashma.model.format import build_prompt, parse_point


@dataclass
class Prediction:
    point: Point | None
    raw: str
    latency_ms: float


class Grounder(Protocol):
    name: str

    def predict(self, image: Image.Image, instruction: str) -> Prediction: ...


def chat_messages(prompt: str, answer: str | None = None) -> list[dict]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


LOCAL_MODEL = "checkpoints/chashma-qwen3vl-2b"
HUB_MODEL = "shricodev/chashmish-babubhaiya"
DEFAULT_MODEL = LOCAL_MODEL if Path(LOCAL_MODEL).exists() else HUB_MODEL
DEFAULT_MAX_PIXELS = 1280 * 800
DEFAULT_LONGEST_EDGE = 1536


def load_processor(
    model_id: str, longest_edge: int | None = None, max_pixels: int | None = None
):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    ip = processor.image_processor
    if hasattr(ip, "merge_size"):
        # qwen's longest_edge is actually a pixel count
        if max_pixels is not None:
            size = {k: v for k, v in dict(ip.size).items() if v is not None}
            ip.size = {**size, "longest_edge": max_pixels}
        # older checkpoints have max_pixels saved in the tokenizer config and it wins over size
        processor.tokenizer.init_kwargs.pop("max_pixels", None)
    elif longest_edge is not None:
        ip.size = {"longest_edge": longest_edge}
    return processor


class VLMGrounder:
    def __init__(
        self,
        model_path: str | Path,
        base_model: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 16,
    ):
        import torch
        from transformers import AutoModelForImageTextToText

        model_path = str(model_path)
        adapter = Path(model_path) / "adapter_config.json"
        base = base_model or model_path
        if adapter.exists() and base_model is None:
            base = json.loads(adapter.read_text())["base_model_name_or_path"]

        meta_path = Path(model_path) / "chashma.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype)
        if adapter.exists():
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_path).merge_and_unload()
        self.model = model.to(self.device).eval()
        saved = ("processor_config.json", "preprocessor_config.json")
        proc_src = model_path if any((Path(model_path) / f).exists() for f in saved) else base
        self.processor = load_processor(
            proc_src,
            longest_edge=meta.get("longest_edge", DEFAULT_LONGEST_EDGE),
            max_pixels=meta.get("max_pixels", DEFAULT_MAX_PIXELS),
        )
        self.max_new_tokens = max_new_tokens
        self.name = Path(model_path).name if Path(model_path).exists() else model_path
        self._torch = torch

    def predict(self, image: Image.Image, instruction: str) -> Prediction:
        start = time.perf_counter()
        prompt = self.processor.apply_chat_template(
            chat_messages(build_prompt(instruction)), add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        text = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return Prediction(parse_point(text), text, (time.perf_counter() - start) * 1000)


API_PROMPT = """\
This is a {w}x{h} screenshot. Where should I click to: {instruction}

Reply with only the pixel coordinates of the click as JSON: {{"x": <int>, "y": <int>}}"""

POINT_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
    "required": ["x", "y"],
}


class APIGrounder:
    def __init__(self, spec: str, max_side: int = 1280):
        from chashma.llm import VisionLLM

        self.llm = VisionLLM(spec, max_tokens=256)
        self.name = spec
        self.max_side = max_side
        self._loop: asyncio.AbstractEventLoop | None = None

    def predict(self, image: Image.Image, instruction: str) -> Prediction:
        # async client sticks to the first loop it ran on, so reuse one
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(self.apredict(image, instruction))

    async def apredict(self, image: Image.Image, instruction: str) -> Prediction:
        from chashma.data.marks import encode_image
        from chashma.llm import parse_json

        start = time.perf_counter()
        scale = min(1.0, self.max_side / max(image.size))
        w, h = round(image.width * scale), round(image.height * scale)
        reply = await self.llm.ask(
            encode_image(image, self.max_side),
            API_PROMPT.format(w=w, h=h, instruction=instruction),
            schema=POINT_SCHEMA,
        )
        point = None
        try:
            data = parse_json(reply.text)
            point = Point(float(data["x"]) / w, float(data["y"]) / h)
        except (ValueError, KeyError, TypeError):
            pass
        return Prediction(point, reply.text, (time.perf_counter() - start) * 1000)


class HTTPGrounder:
    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.name = self.base_url
        self.timeout_s = timeout_s

    def _request(self, image: Image.Image, instruction: str) -> dict:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return {
            "url": f"{self.base_url}/v1/ground",
            "files": {"image": ("screenshot.png", buf.getvalue(), "image/png")},
            "data": {"instruction": instruction},
        }

    @staticmethod
    def _parse(resp, started: float) -> Prediction:
        latency = (time.perf_counter() - started) * 1000
        if resp.status_code == 502:
            return Prediction(None, resp.text, latency)
        resp.raise_for_status()
        body = resp.json()
        return Prediction(Point(body["x_norm"], body["y_norm"]), body["raw"], latency)

    def predict(self, image: Image.Image, instruction: str) -> Prediction:
        import httpx

        start = time.perf_counter()
        resp = httpx.post(**self._request(image, instruction), timeout=self.timeout_s)
        return self._parse(resp, start)

    async def apredict(self, image: Image.Image, instruction: str) -> Prediction:
        import httpx

        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(**self._request(image, instruction))
        return self._parse(resp, start)


def load_grounder(spec: str, **kwargs) -> Grounder:
    if spec.startswith(("anthropic:", "openai:")):
        return APIGrounder(spec)
    if spec.startswith(("http://", "https://")):
        return HTTPGrounder(spec)
    return VLMGrounder(spec, **kwargs)
