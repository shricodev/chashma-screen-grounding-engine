from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from pydantic import BaseModel

from chashma.data.crawler import iter_captures
from chashma.data.marks import draw_marks, encode_image, select_targets
from chashma.data.schema import Element, PageCapture
from chashma.llm import Usage, VisionLLM, parse_json

log = logging.getLogger(__name__)

STYLES = ("direct", "functional", "visual")

PROMPT = """\
This screenshot is from {url} ({viewport} viewport). Numbered colored boxes mark \
clickable elements.

For each numbered mark, write three instructions a person might give to an assistant \
that controls their computer. Each one must lead to clicking exactly that element.
- "direct": refers to the element by its visible text or label, e.g. "click Sign in"
- "functional": states the goal without quoting the label, e.g. "log into my account"
- "visual": describes what it looks like and/or where it is, e.g. "the magnifying \
glass icon at the top right"

Rules:
- Each instruction must point to that one element and no other element on screen.
- Describe only what a person can see on the page. Never mention the boxes, the \
numbers or the colors of the marks.
- Write like a real user: short, natural, varied, under 15 words.
- If a mark does not sit on a real control or its purpose is unclear, set "skip": true.

Hints from the page HTML (may be imperfect):
{hints}

Reply with JSON only:
{{"marks": [{{"id": 1, "skip": false, "direct": "...", "functional": "...", "visual": "..."}}]}}
"""

_LEAKS_OVERLAY = re.compile(
    r"\b(box(ed)?|marks?|marked|numbered|outlined|highlighted|label(ed)? \d)\b|#\s?\d|\(\d+\)",
    re.IGNORECASE,
)
MAX_INSTRUCTION_CHARS = 160


class Instruction(BaseModel):
    element_idx: int
    style: str
    text: str


class Annotation(BaseModel):
    capture_id: str
    model: str
    instructions: list[Instruction]
    skipped: list[int]
    input_tokens: int
    output_tokens: int


def annotation_path(capture_dir: Path, capture: PageCapture) -> Path:
    return capture_dir / capture.domain / f"{capture.id}.instr.json"


def _hint(n: int, el: Element) -> str:
    parts = [f"{n}: {el.role}"]
    if el.name:
        parts.append(f'name="{el.name}"')
    if el.href:
        parts.append(f"href={urlparse(el.href).path or '/'}")
    if el.is_icon:
        parts.append("icon")
    return ", ".join(parts)


def clean_instruction(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    text = " ".join(text.split()).strip().strip('"')
    if not text or len(text) > MAX_INSTRUCTION_CHARS or _LEAKS_OVERLAY.search(text):
        return None
    return text


def parse_annotation(reply: dict, targets: list[Element]) -> tuple[list[Instruction], list[int]]:
    instructions, skipped = [], []
    for mark in reply.get("marks", []):
        try:
            n = int(mark["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 1 <= n <= len(targets):
            continue
        el = targets[n - 1]
        if mark.get("skip"):
            skipped.append(el.idx)
            continue
        for style in STYLES:
            if (text := clean_instruction(mark.get(style))) is not None:
                instructions.append(Instruction(element_idx=el.idx, style=style, text=text))
    return instructions, skipped


async def annotate_capture(
    llm: VisionLLM, capture: PageCapture, capture_dir: Path, marks_per_page: int, seed: int
) -> Annotation | None:
    rng = random.Random(f"{seed}:{capture.id}")
    targets = select_targets(capture.elements, k=marks_per_page, rng=rng)
    if not targets:
        return None
    image = Image.open(capture_dir / capture.domain / capture.screenshot)
    marked = encode_image(draw_marks(image, targets))
    prompt = PROMPT.format(
        url=capture.url,
        viewport=capture.viewport.name,
        hints="\n".join(_hint(n, el) for n, el in enumerate(targets, start=1)),
    )
    reply = await llm.ask(marked, prompt, json_mode=True)
    instructions, skipped = parse_annotation(parse_json(reply.text), targets)
    return Annotation(
        capture_id=capture.id,
        model=llm.spec,
        instructions=instructions,
        skipped=skipped,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
    )


@dataclass
class AnnotateStats:
    captures: int = 0
    instructions: int = 0
    failed: int = 0
    usage: Usage | None = None


async def annotate_all(
    capture_dir: Path,
    model: str,
    marks_per_page: int = 8,
    concurrency: int = 4,
    limit: int | None = None,
    seed: int = 0,
) -> AnnotateStats:
    llm = VisionLLM(model)
    stats = AnnotateStats(usage=llm.usage)
    todo = [c for c in iter_captures(capture_dir) if not annotation_path(capture_dir, c).exists()]
    if limit is not None:
        todo = todo[:limit]
    sem = asyncio.Semaphore(concurrency)

    async def one(capture: PageCapture) -> None:
        async with sem:
            try:
                ann = await annotate_capture(llm, capture, capture_dir, marks_per_page, seed)
            except Exception as e:
                stats.failed += 1
                log.warning("annotation failed for %s: %s", capture.id, e)
                return
        if ann is None:
            return
        annotation_path(capture_dir, capture).write_text(ann.model_dump_json(indent=1))
        stats.captures += 1
        stats.instructions += len(ann.instructions)

    await asyncio.gather(*(one(c) for c in todo))
    return stats


def load_annotation(capture_dir: Path, capture: PageCapture) -> Annotation | None:
    path = annotation_path(capture_dir, capture)
    if not path.exists():
        return None
    return Annotation.model_validate(json.loads(path.read_text()))
