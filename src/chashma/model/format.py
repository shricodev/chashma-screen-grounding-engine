from __future__ import annotations

import re

from chashma.geometry import Point

GRID = 1000

PROMPT = "Where should I click to: {instruction}\nAnswer with the point."

_POINT = re.compile(r"<point>\s*(\d{1,4})\s*,\s*(\d{1,4})\s*</point>")
_LOOSE = re.compile(r"(\d{1,4})\s*,\s*(\d{1,4})")


def build_prompt(instruction: str) -> str:
    return PROMPT.format(instruction=instruction.strip())


def encode_point(p: Point) -> str:
    x = min(GRID - 1, max(0, int(p.x * GRID)))
    y = min(GRID - 1, max(0, int(p.y * GRID)))
    return f"<point>{x},{y}</point>"


def parse_point(text: str) -> Point | None:
    match = _POINT.search(text) or _LOOSE.search(text)
    if not match:
        return None
    x, y = int(match.group(1)), int(match.group(2))
    if x >= GRID or y >= GRID:
        return None
    return Point((x + 0.5) / GRID, (y + 0.5) / GRID)
