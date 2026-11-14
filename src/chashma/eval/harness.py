from __future__ import annotations

import asyncio
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image
from rich.progress import track

from chashma.data.schema import GroundingSample
from chashma.geometry import Box
from chashma.model.grounder import Grounder, Prediction


@dataclass
class EvalExample:
    id: str
    instruction: str
    box: Box
    load_image: Callable[[], Image.Image]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalRecord:
    id: str
    instruction: str
    hit: bool
    parsed: bool
    distance: float | None
    latency_ms: float
    raw: str
    tags: dict[str, str]


@dataclass
class EvalReport:
    model: str
    dataset: str
    n: int
    accuracy: float
    parse_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    breakdown: dict[str, dict[str, dict[str, float]]]
    records: list[EvalRecord]

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("records")
        return d


def examples_from_samples(
    samples: Iterable[GroundingSample], image_root: Path
) -> list[EvalExample]:
    def loader(path: Path) -> Callable[[], Image.Image]:
        return lambda: Image.open(path).convert("RGB")

    return [
        EvalExample(
            id=s.id,
            instruction=s.instruction,
            box=s.box,
            load_image=loader(image_root / s.image),
            tags={
                "style": s.style,
                "type": "icon" if s.is_icon else "text",
                "viewport": s.viewport,
                "theme": s.color_scheme,
            },
        )
        for s in samples
    ]


def score(example: EvalExample, pred: Prediction) -> EvalRecord:
    hit, dist = False, None
    if pred.point is not None:
        hit = example.box.contains(pred.point)
        c = example.box.center
        dist = math.hypot(pred.point.x - c.x, pred.point.y - c.y)
    return EvalRecord(
        id=example.id,
        instruction=example.instruction,
        hit=hit,
        parsed=pred.point is not None,
        distance=dist,
        latency_ms=pred.latency_ms,
        raw=pred.raw,
        tags=example.tags,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize(model: str, dataset: str, records: list[EvalRecord]) -> EvalReport:
    n = len(records)
    groups: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        for key, value in r.tags.items():
            groups[key][value].append(r.hit)
    breakdown = {
        key: {
            value: {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4)}
            for value, hits in sorted(values.items())
        }
        for key, values in groups.items()
    }
    latencies = [r.latency_ms for r in records if r.latency_ms > 0]
    return EvalReport(
        model=model,
        dataset=dataset,
        n=n,
        accuracy=round(sum(r.hit for r in records) / n, 4) if n else 0.0,
        parse_rate=round(sum(r.parsed for r in records) / n, 4) if n else 0.0,
        latency_p50_ms=round(statistics.median(latencies), 1) if latencies else 0.0,
        latency_p95_ms=round(_percentile(latencies, 0.95), 1),
        breakdown=breakdown,
        records=records,
    )


def evaluate(
    grounder: Grounder, examples: list[EvalExample], dataset: str, concurrency: int = 8
) -> EvalReport:
    if hasattr(grounder, "apredict"):
        records = asyncio.run(_evaluate_async(grounder, examples, concurrency))
    else:
        records = [
            score(ex, grounder.predict(ex.load_image(), ex.instruction))
            for ex in track(examples, description=f"eval {grounder.name}")
        ]
    return summarize(grounder.name, dataset, records)


async def _evaluate_async(grounder, examples: list[EvalExample], concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def one(ex: EvalExample) -> EvalRecord:
        async with sem:
            try:
                pred = await grounder.apredict(ex.load_image(), ex.instruction)
            except Exception as e:
                pred = Prediction(None, f"error: {e}", 0.0)
            return score(ex, pred)

    return await asyncio.gather(*(one(ex) for ex in examples))


def write_report(report: EvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.dataset}__{report.model}".replace("/", "_").replace(":", "_")
    (out_dir / f"{stem}.json").write_text(json.dumps(report.summary(), indent=2))
    with (out_dir / f"{stem}.records.jsonl").open("w") as f:
        for r in report.records:
            f.write(json.dumps(asdict(r)) + "\n")
    return out_dir / f"{stem}.json"
