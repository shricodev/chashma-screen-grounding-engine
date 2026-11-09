from __future__ import annotations

from chashma.eval.harness import EvalExample
from chashma.geometry import Box

HF_DATASET = "rootsautomation/ScreenSpot"

PLATFORM_GROUPS = {
    "web": {"tool", "shop", "gitlab", "forum"},
    "mobile": {"ios", "android"},
    "desktop": {"macos", "windows"},
}
SOURCE_TO_GROUP = {src: group for group, srcs in PLATFORM_GROUPS.items() for src in srcs}


def resolve_platforms(platforms: list[str] | None) -> set[str] | None:
    if not platforms:
        return None
    sources: set[str] = set()
    for p in platforms:
        if p in PLATFORM_GROUPS:
            sources |= PLATFORM_GROUPS[p]
        elif p in SOURCE_TO_GROUP:
            sources.add(p)
        else:
            valid = sorted({*PLATFORM_GROUPS, *SOURCE_TO_GROUP})
            raise ValueError(f"unknown ScreenSpot platform {p!r}, expected one of {valid}")
    return sources


def load_screenspot(
    split: str = "test", platforms: list[str] | None = None, limit: int | None = None
) -> list[EvalExample]:
    from datasets import load_dataset

    sources = resolve_platforms(platforms)
    ds = load_dataset(HF_DATASET, split=split)
    meta = ds.select_columns(["data_source", "instruction", "bbox", "data_type"])
    examples = []
    for i, row in enumerate(meta):
        source = row["data_source"]
        if sources is not None and source not in sources:
            continue
        examples.append(
            EvalExample(
                id=f"screenspot-{i}",
                instruction=row["instruction"],
                box=_lazy_box(ds, i, row["bbox"]),
                load_image=(lambda i=i: ds[i]["image"].convert("RGB")),
                tags={
                    "platform": SOURCE_TO_GROUP.get(source, source),
                    "source": source,
                    "type": row["data_type"],
                },
            )
        )
        if limit and len(examples) >= limit:
            break
    return examples


def _lazy_box(ds, i: int, bbox: list[float]) -> Box:
    if max(float(v) for v in bbox) <= 1.0:
        return _to_box(bbox, 1, 1)
    return _to_box(bbox, *ds[i]["image"].size)


def _to_box(bbox: list[float], w: int, h: int) -> Box:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if max(x1, y1, x2, y2) <= 1.0:
        return Box(x1, y1, x2, y2)
    return Box(x1 / w, y1 / h, x2 / w, y2 / h)
