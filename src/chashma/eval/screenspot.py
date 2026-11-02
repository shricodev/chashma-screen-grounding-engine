from __future__ import annotations

from chashma.eval.harness import EvalExample
from chashma.geometry import Box

HF_DATASET = "rootsautomation/ScreenSpot"


def load_screenspot(split: str = "test", platforms: list[str] | None = None,
                    limit: int | None = None) -> list[EvalExample]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split=split)
    examples = []
    for i, row in enumerate(ds):
        source = row["data_source"]
        if platforms and source not in platforms:
            continue
        image = row["image"]
        w, h = image.size
        box = _to_box(row["bbox"], w, h)
        examples.append(
            EvalExample(
                id=f"screenspot-{i}",
                instruction=row["instruction"],
                box=box,
                load_image=(lambda img=image: img.convert("RGB")),
                tags={"platform": source, "type": row["data_type"]},
            )
        )
        if limit and len(examples) >= limit:
            break
    return examples


def _to_box(bbox: list[float], w: int, h: int) -> Box:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if max(x1, y1, x2, y2) <= 1.0:
        return Box(x1, y1, x2, y2)
    return Box(x1 / w, y1 / h, x2 / w, y2 / h)
