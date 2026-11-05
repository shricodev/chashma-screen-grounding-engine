from __future__ import annotations

import json
import shutil
from pathlib import Path

DEFAULT_BASE = "HuggingFaceTB/SmolVLM-256M-Instruct"
SKIP = ("checkpoint-", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state")


def checkpoint_meta(model_path: Path) -> dict:
    path = model_path / "chashma.json"
    return json.loads(path.read_text()) if path.exists() else {}


def export_checkpoint(model_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    adapter = model_path / "adapter_config.json"
    if adapter.exists():
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        base = json.loads(adapter.read_text())["base_model_name_or_path"]
        model = AutoModelForImageTextToText.from_pretrained(base)
        model = PeftModel.from_pretrained(model, str(model_path)).merge_and_unload()
        model.save_pretrained(out_dir)
        AutoProcessor.from_pretrained(str(model_path)).save_pretrained(out_dir)
        return
    for item in model_path.iterdir():
        if item.is_file() and not item.name.startswith(SKIP):
            shutil.copy2(item, out_dir / item.name)


def _results_table(reports_dir: Path) -> str:
    rows = []
    for path in sorted(reports_dir.glob("*.json")):
        r = json.loads(path.read_text())
        rows.append(
            f"| {r['model']} | {r['dataset']} | {r['n']} | {r['accuracy']:.1%} "
            f"| {r['latency_p50_ms']:.0f} ms |"
        )
    if not rows:
        return "_No evaluation reports found._"
    header = "| model | eval set | n | click accuracy | p50 latency |\n|---|---|---|---|---|"
    return "\n".join([header, *rows])


def model_card(
    repo_id: str, base_model: str, meta: dict, reports_dir: Path, source_url: str | None = None
) -> str:
    name = repo_id.split("/")[-1]
    project = f"[chashma]({source_url})" if source_url else "chashma"
    trained_on = meta.get("train_samples")
    data_line = (
        f"{trained_on:,} (screenshot, instruction, box) examples" if trained_on
        else "(screenshot, instruction, box) examples"
    )
    return f"""---
license: apache-2.0
base_model: {base_model}
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- gui-grounding
- computer-use
- web-agents
---

# {name}

A small GUI grounding model: give it a screenshot and an instruction such as
*"open the pricing page"* and it answers with where to click, as
`<point>x,y</point>` on a 0..999 grid relative to the image.

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}) on {data_line}
built by {project}: real websites rendered with
Playwright, click targets taken from the DOM, instructions written by a vision LLM.
Trained on a single 6 GB laptop GPU.

## Usage

```python
from PIL import Image
from chashma import load_grounder

grounder = load_grounder("{repo_id}")
pred = grounder.predict(Image.open("screenshot.png"), "open the pricing page")
x, y = pred.point.to_pixels(*Image.open("screenshot.png").size)
```

Or with plain transformers, prompt with
`Where should I click to: <instruction>\\nAnswer with the point.` and parse the
`<point>x,y</point>` reply.

## Results

Click accuracy = the predicted point lands inside the target element's box.

{_results_table(reports_dir)}

## Limitations

- Trained on desktop and mobile *web* pages in English; native mobile and desktop apps
  are out of distribution.
- Small targets on large screens are the main failure mode.
- It predicts a location only. It has no notion of whether clicking is safe or
  intended; keep a human or a policy layer in the loop for agents acting on real
  accounts.
"""


def publish(
    model_path: Path,
    repo_id: str,
    reports_dir: Path,
    out_root: Path = Path("dist/hf"),
    push: bool = False,
    private: bool = False,
    source_url: str | None = None,
) -> Path | str:
    meta = checkpoint_meta(model_path)
    out_dir = out_root / repo_id.split("/")[-1]
    export_checkpoint(model_path, out_dir)
    card = model_card(
        repo_id, meta.get("base_model", DEFAULT_BASE), meta, reports_dir, source_url
    )
    (out_dir / "README.md").write_text(card)
    if not push:
        return out_dir

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(out_dir), repo_id=repo_id, commit_message="upload chashma checkpoint"
    )
    return commit.commit_url
