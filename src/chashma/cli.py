from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chashma.model.grounder import DEFAULT_MODEL

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

CAPTURES = Path("data/captures")
DATASET = Path("data/dataset")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def crawl(
    seeds: Path = typer.Option(Path("configs/seeds.txt"), help="URL list or tranco csv"),
    out: Path = typer.Option(CAPTURES),
    limit: int | None = typer.Option(None, help="only the first N seeds"),
    variants: int = typer.Option(3, help="captures per page (viewport/theme/scroll combos)"),
    links: int = typer.Option(3, help="extra same-site pages per seed"),
    concurrency: int = typer.Option(4),
    ignore_robots: bool = typer.Option(False, help="don't check robots.txt"),
) -> None:
    """Render pages and save screenshots + clickable element boxes."""
    from chashma.data.crawler import CrawlConfig, Crawler, load_seeds

    urls = load_seeds(seeds, limit)
    cfg = CrawlConfig(
        out_dir=out,
        variants_per_page=variants,
        links_per_site=links,
        concurrency=concurrency,
        respect_robots=not ignore_robots,
    )
    console.print(f"crawling {len(urls)} seeds -> {out}")
    stats = asyncio.run(Crawler(cfg).run(urls))
    console.print(stats)


@app.command()
def annotate(
    captures: Path = typer.Option(CAPTURES),
    model: str = typer.Option("anthropic:claude-haiku-5-5", help="provider:model"),
    marks: int = typer.Option(8, help="elements marked per screenshot"),
    concurrency: int = typer.Option(4),
    limit: int | None = typer.Option(None, help="annotate at most N captures"),
) -> None:
    """Write natural-language instructions for marked elements with a vision LLM."""
    from chashma.data.annotate import annotate_all

    stats = asyncio.run(annotate_all(captures, model, marks, concurrency, limit))
    u = stats.usage
    console.print(
        f"annotated {stats.captures} captures, {stats.instructions} instructions, "
        f"{stats.failed} failed | {u.calls} calls, {u.input_tokens:,} in / "
        f"{u.output_tokens:,} out tokens"
    )


@app.command()
def build(
    captures: Path = typer.Option(CAPTURES),
    out: Path = typer.Option(DATASET),
    templates: int = typer.Option(-1, help="template instructions per capture, -1 = all"),
    seed: int = typer.Option(0),
) -> None:
    """Assemble captures + annotations into train/val/test splits."""
    from chashma.data.build import build_dataset

    stats = build_dataset(
        captures, out, templates_per_capture=templates if templates >= 0 else None, seed=seed
    )
    table = Table("metric", "value")
    table.add_row("captures", str(stats.captures))
    table.add_row("near duplicates dropped", str(stats.near_duplicates))
    table.add_row("domains", str(len(stats.domains)))
    for split, n in sorted(stats.samples.items()):
        table.add_row(f"samples/{split}", str(n))
    for style, n in sorted(stats.styles.items()):
        table.add_row(f"style/{style}", str(n))
    table.add_row("icon targets", str(stats.icons))
    console.print(table)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/train_qwen3vl_2b.yaml")),
    max_train_samples: int | None = typer.Option(None),
    epochs: float | None = typer.Option(None),
    output_dir: Path | None = typer.Option(None),
    resume: bool = typer.Option(False, help="continue from the last saved checkpoint"),
) -> None:
    """Fine-tune a small VLM on the built dataset."""
    from chashma.train.sft import TrainConfig
    from chashma.train.sft import train as run_train

    logging.getLogger("chashma").setLevel(logging.INFO)
    cfg = TrainConfig.from_yaml(
        config,
        max_train_samples=max_train_samples,
        epochs=epochs,
        output_dir=output_dir,
        resume=resume or None,
    )
    out = run_train(cfg)
    console.print(f"saved to {out}")


@app.command("eval")
def eval_cmd(
    model: str = typer.Option(DEFAULT_MODEL, help="checkpoint dir, HF id, or provider:model"),
    dataset: str = typer.Option("chashma", help="'chashma' (our test split) or 'screenspot'"),
    data_dir: Path = typer.Option(DATASET),
    split: str = typer.Option("test"),
    platform: list[str] | None = typer.Option(None, help="screenspot platforms, e.g. web"),
    limit: int | None = typer.Option(None, help="seeded random subset of this size"),
    concurrency: int = typer.Option(8, help="parallel requests for API models"),
    out: Path = typer.Option(Path("reports")),
) -> None:
    """Measure click accuracy and write a report."""
    from chashma.eval.harness import evaluate, examples_from_samples, write_report
    from chashma.model.grounder import load_grounder

    name = dataset
    if dataset == "screenspot":
        from chashma.eval.screenspot import load_screenspot

        try:
            examples = load_screenspot(platforms=platform)
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--platform") from e
        if platform:
            name = f"screenspot-{'+'.join(sorted(platform))}"
    else:
        from chashma.data.build import load_samples

        samples, root = load_samples(data_dir, split)
        examples = examples_from_samples(samples, root)
        name = f"chashma-{split}"
    if not examples:
        console.print("[red]no examples matched[/red]")
        raise typer.Exit(1)
    if limit and limit < len(examples):
        examples = random.Random(0).sample(examples, limit)
        name = f"{name}-n{limit}"

    report = evaluate(load_grounder(model), examples, name, concurrency=concurrency)
    path = write_report(report, out)

    table = Table(f"{report.model} on {dataset}", "n", "accuracy")
    table.add_row("overall", str(report.n), f"{report.accuracy:.1%}")
    for key, values in report.breakdown.items():
        for value, m in values.items():
            table.add_row(f"  {key}={value}", str(m["n"]), f"{m['accuracy']:.1%}")
    console.print(table)
    console.print(
        f"parse rate {report.parse_rate:.1%} | latency p50 {report.latency_p50_ms:.0f}ms "
        f"p95 {report.latency_p95_ms:.0f}ms | report: {path}"
    )


@app.command()
def predict(
    image: Path = typer.Argument(..., exists=True),
    instruction: str = typer.Argument(...),
    model: str = typer.Option(DEFAULT_MODEL),
    save: Path | None = typer.Option(None, help="write the screenshot with the click drawn"),
) -> None:
    """Ground one instruction on one screenshot."""
    from PIL import Image, ImageDraw

    from chashma.model.grounder import load_grounder

    img = Image.open(image).convert("RGB")
    pred = load_grounder(model).predict(img, instruction)
    if pred.point is None:
        console.print(f"[red]could not parse a point from:[/red] {pred.raw!r}")
        raise typer.Exit(1)
    x, y = pred.point.to_pixels(*img.size)
    console.print(f"click at ({x}, {y})  raw={pred.raw!r}  {pred.latency_ms:.0f}ms")
    if save:
        draw = ImageDraw.Draw(img)
        r = max(6, img.width // 120)
        draw.ellipse([x - r, y - r, x + r, y + r], outline="#ff1744", width=4)
        draw.line([x - 2 * r, y, x + 2 * r, y], fill="#ff1744", width=2)
        draw.line([x, y - 2 * r, x, y + 2 * r], fill="#ff1744", width=2)
        img.save(save)


@app.command()
def serve(
    model: str = typer.Option(DEFAULT_MODEL),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the grounding API."""
    import uvicorn

    from chashma.serve.app import create_app

    uvicorn.run(create_app(model), host=host, port=port)


@app.command()
def publish(
    repo: str = typer.Option(..., help="Hugging Face repo id, e.g. user/chashma-qwen3vl-2b"),
    model: Path = typer.Option(Path(DEFAULT_MODEL), exists=True, file_okay=False),
    reports: Path = typer.Option(Path("reports"), help="eval reports for the model card"),
    source_url: str | None = typer.Option(None, help="link to the code, shown on the card"),
    push: bool = typer.Option(False, help="upload; without it only dist/hf/ is prepared"),
    private: bool = typer.Option(False),
) -> None:
    """Prepare (and optionally upload) a checkpoint + model card for the Hugging Face Hub."""
    from chashma.publish import publish as run_publish

    result = run_publish(
        model, repo, reports, push=push, private=private, source_url=source_url
    )
    if push:
        console.print(f"uploaded: {result}")
    else:
        console.print(f"prepared {result} (review it, then rerun with --push to upload)")


@app.command()
def agent(
    task: str = typer.Argument(..., help="what the agent should do"),
    url: str = typer.Option(..., help="start page"),
    grounder: str = typer.Option(
        DEFAULT_MODEL, help="checkpoint, http://host:port, or provider:model"
    ),
    planner: str = typer.Option("anthropic:claude-sonnet-5-5", help="provider:model"),
    max_steps: int = typer.Option(15),
    headed: bool = typer.Option(False, help="show the browser window"),
    out: Path | None = typer.Option(None, help="trajectory dir, default runs/agent/<time>"),
) -> None:
    """Run a browser task: the planner decides what to click, chashma finds where."""
    from datetime import datetime

    from chashma.agent.browser import BrowserAgent
    from chashma.llm import VisionLLM
    from chashma.model.grounder import load_grounder

    logging.getLogger("chashma").setLevel(logging.INFO)
    out = out or Path("runs/agent") / datetime.now().strftime("%Y%m%d-%H%M%S")
    runner = BrowserAgent(
        VisionLLM(planner, max_tokens=8000),
        load_grounder(grounder),
        out,
        max_steps=max_steps,
        headless=not headed,
    )
    result = asyncio.run(runner.run(task, url))

    for step in result.steps:
        console.print(step.describe())
    status = "[green]done[/green]" if result.finished else "[yellow]gave up[/yellow]"
    console.print(f"{status}: {result.answer or '-'}")
    console.print(
        f"planner tokens {result.planner_input_tokens:,} in / "
        f"{result.planner_output_tokens:,} out | trajectory: {out}"
    )


if __name__ == "__main__":
    app()
