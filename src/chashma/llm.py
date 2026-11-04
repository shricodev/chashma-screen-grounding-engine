from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1


@dataclass
class VisionReply:
    text: str
    input_tokens: int
    output_tokens: int


class VisionLLM:
    def __init__(self, spec: str, max_tokens: int = 2048):
        provider, _, model = spec.partition(":")
        if provider not in ("anthropic", "openai") or not model:
            raise ValueError(f"expected 'anthropic:<model>' or 'openai:<model>', got {spec!r}")
        self.spec, self.provider, self.model = spec, provider, model
        self.max_tokens = max_tokens
        self.usage = Usage()
        if provider == "anthropic":
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(max_retries=4)
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(max_retries=4)

    async def ask(self, image: bytes, prompt: str, media_type: str = "image/jpeg",
                  json_mode: bool = False) -> VisionReply:
        b64 = base64.b64encode(image).decode()
        if self.provider == "anthropic":
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            reply = VisionReply(text, resp.usage.input_tokens, resp.usage.output_tokens)
        else:
            extra = {"response_format": {"type": "json_object"}} if json_mode else {}
            resp = await self._client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self.max_tokens,
                **extra,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = resp.choices[0].message.content or ""
            u = resp.usage
            reply = VisionReply(text, u.prompt_tokens if u else 0, u.completion_tokens if u else 0)
        self.usage.add(reply.input_tokens, reply.output_tokens)
        return reply


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict:
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON object in reply")
    return json.loads(match.group(0))
