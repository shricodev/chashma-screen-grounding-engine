from __future__ import annotations

import re

from chashma.geometry import Point

GRID = 1000

PROMPT = (
    "Point to the UI element I should click to: {instruction}. "
    "Answer with the click point as (x, y)."
)

_POINT = re.compile(r"\(\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\)")
_LOOSE = re.compile(r"(\d{1,4})\s*,\s*(\d{1,4})")


def build_prompt(instruction: str) -> str:
    return PROMPT.format(instruction=instruction.strip().rstrip("."))


def encode_point(p: Point) -> str:
    x = min(GRID - 1, max(0, int(p.x * GRID)))
    y = min(GRID - 1, max(0, int(p.y * GRID)))
    return f"({x}, {y})"


def parse_point(text: str) -> Point | None:
    match = _POINT.search(text) or _LOOSE.search(text)
    if not match:
        return None
    x, y = int(match.group(1)), int(match.group(2))
    if x > GRID or y > GRID:
        return None
    x, y = min(x, GRID - 1), min(y, GRID - 1)
    return Point((x + 0.5) / GRID, (y + 0.5) / GRID)
