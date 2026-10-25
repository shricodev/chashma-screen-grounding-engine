from __future__ import annotations

import io
import random

from PIL import Image, ImageDraw, ImageFont

from chashma.data.schema import Element
from chashma.geometry import Box

MARK_COLORS = ["#ff1744", "#2979ff", "#00c853", "#ff9100", "#d500f9", "#00b8d4", "#ffd600"]


def _overlaps(a: Box, b: Box, margin: float = 0.01) -> bool:
    return not (
        a.x2 + margin < b.x1 or b.x2 + margin < a.x1 or a.y2 + margin < b.y1 or b.y2 + margin < a.y1
    )


def select_targets(
    elements: list[Element], k: int = 8, rng: random.Random | None = None, min_area: float = 1e-4
) -> list[Element]:
    rng = rng or random.Random(0)
    candidates = [e for e in elements if e.box.area >= min_area]
    buckets: dict[str, list[Element]] = {}
    for el in candidates:
        kind = "icon" if el.is_icon else ("input" if el.role in ("textbox", "searchbox",
                                                                  "combobox") else "text")
        buckets.setdefault(kind, []).append(el)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    picked: list[Element] = []
    kinds = list(buckets)
    while len(picked) < k and any(buckets.values()):
        for kind in kinds:
            while buckets[kind]:
                el = buckets[kind].pop()
                if all(not _overlaps(el.box, p.box) for p in picked):
                    picked.append(el)
                    break
            if len(picked) >= k:
                break
    return picked


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_marks(image: Image.Image, targets: list[Element]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    font = _font(max(12, w // 80))
    for n, el in enumerate(targets, start=1):
        color = MARK_COLORS[(n - 1) % len(MARK_COLORS)]
        x1, y1, x2, y2 = el.box.to_pixels(w, h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(n)
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        lx = x1
        ly = y1 - th - 6 if y1 - th - 6 >= 0 else y1 + 2
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=color)
        draw.text((lx + 3, ly + 1), label, fill="white", font=font)
    return out


def encode_image(image: Image.Image, max_side: int = 1280, fmt: str = "JPEG") -> bytes:
    img = image.convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    return buf.getvalue()
