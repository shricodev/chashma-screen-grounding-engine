from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from chashma.data.annotate import load_annotation
from chashma.data.schema import GroundingSample, PageCapture, iter_captures
from chashma.data.templates import template_instructions


def split_for(domain: str, val_pct: int = 5, test_pct: int = 5) -> str:
    bucket = int(hashlib.sha1(domain.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def dhash(image: Image.Image, size: int = 8) -> int:
    img = image.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(img.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


@dataclass
class BuildStats:
    captures: int = 0
    near_duplicates: int = 0
    samples: Counter = field(default_factory=Counter)
    styles: Counter = field(default_factory=Counter)
    icons: int = 0
    domains: Counter = field(default_factory=Counter)


def _samples_for(
    capture: PageCapture, capture_dir: Path, split: str, rng: random.Random, templates: int
) -> list[GroundingSample]:
    by_idx = {e.idx: e for e in capture.elements}
    pairs: list[tuple[int, str, str, str]] = []

    if (ann := load_annotation(capture_dir, capture)) is not None:
        pairs += [(i.element_idx, i.style, ann.model, i.text) for i in ann.instructions]
    for el, text in template_instructions(
        capture.elements, rng, mobile=capture.viewport.is_mobile, k=templates
    ):
        pairs.append((el.idx, "template", "template", text))

    seen = set()
    samples = []
    for idx, style, source, text in pairs:
        el = by_idx.get(idx)
        key = (idx, text.lower())
        if el is None or key in seen:
            continue
        seen.add(key)
        samples.append(
            GroundingSample(
                id=hashlib.sha1(f"{capture.id}|{idx}|{text}".encode()).hexdigest()[:16],
                image=f"{capture.domain}/{capture.screenshot}",
                image_size=capture.image_size,
                instruction=text,
                bbox=el.bbox,
                style=style,  # type: ignore[arg-type]
                source=source,
                capture_id=capture.id,
                domain=capture.domain,
                url=capture.url,
                viewport=capture.viewport.name,
                color_scheme=capture.color_scheme,
                element_role=el.role,
                is_icon=el.is_icon,
                split=split,  # type: ignore[arg-type]
            )
        )
    return samples


def build_dataset(
    capture_dir: Path,
    out_dir: Path,
    templates_per_capture: int = 4,
    val_pct: int = 5,
    test_pct: int = 5,
    seed: int = 0,
    max_hash_distance: int = 2,
) -> BuildStats:
    stats = BuildStats()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {s: (out_dir / f"{s}.jsonl").open("w") for s in ("train", "val", "test")}
    hashes: dict[str, list[int]] = {}
    try:
        for capture in iter_captures(capture_dir):
            with Image.open(capture_dir / capture.domain / capture.screenshot) as img:
                h = dhash(img)
            prev = hashes.setdefault(capture.url, [])
            if any(bin(h ^ p).count("1") <= max_hash_distance for p in prev):
                stats.near_duplicates += 1
                continue
            prev.append(h)

            stats.captures += 1
            split = split_for(capture.domain, val_pct, test_pct)
            rng = random.Random(f"{seed}:{capture.id}")
            for sample in _samples_for(capture, capture_dir, split, rng, templates_per_capture):
                files[split].write(sample.model_dump_json() + "\n")
                stats.samples[split] += 1
                stats.styles[sample.style] += 1
                stats.icons += sample.is_icon
                stats.domains[sample.domain] += 1
    finally:
        for f in files.values():
            f.close()

    # not asdict, it messes up the Counters
    meta = {
        "image_root": str(capture_dir.resolve()),
        "seed": seed,
        "stats": {
            "captures": stats.captures,
            "near_duplicates": stats.near_duplicates,
            "samples": dict(stats.samples),
            "styles": dict(stats.styles),
            "icons": stats.icons,
            "domains": len(stats.domains),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return stats


def load_samples(
    dataset_dir: Path, split: str, image_root: Path | None = None
) -> tuple[list[GroundingSample], Path]:
    meta = json.loads((dataset_dir / "meta.json").read_text())
    with (dataset_dir / f"{split}.jsonl").open() as f:
        samples = [GroundingSample.model_validate_json(line) for line in f if line.strip()]
    for root in (image_root, Path(meta["image_root"]), dataset_dir.parent / "captures"):
        if root is not None and root.is_dir():
            return samples, root
    raise FileNotFoundError(f"image root not found for {dataset_dir}, pass image_root")
