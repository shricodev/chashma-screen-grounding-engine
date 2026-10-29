from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

CAPTURES = Path("data/captures")
DATASET = Path("data/dataset")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
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
    templates: int = typer.Option(4, help="template instructions per capture"),
    seed: int = typer.Option(0),
) -> None:
    """Assemble captures + annotations into train/val/test splits."""
    from chashma.data.build import build_dataset

    stats = build_dataset(captures, out, templates_per_capture=templates, seed=seed)
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


if __name__ == "__main__":
    app()
