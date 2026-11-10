from __future__ import annotations

import json
import shutil
from pathlib import Path

DEFAULT_BASE = "Qwen/Qwen3-VL-2B-Instruct"
SKIP = ("checkpoint-", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state")


def checkpoint_meta(model_path: Path) -> dict:
    path = model_path / "chashma.json"
    return json.loads(path.read_text()) if path.exists() else {}


def export_checkpoint(model_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    meta = checkpoint_meta(model_path)
    adapter = model_path / "adapter_config.json"
    if not adapter.exists():
        for item in model_path.iterdir():
            if item.is_file() and not item.name.startswith(SKIP):
                shutil.copy2(item, out_dir / item.name)
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    from chashma.model.grounder import DEFAULT_MAX_PIXELS, load_processor

    base = json.loads(adapter.read_text())["base_model_name_or_path"]
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(model_path)).merge_and_unload()
    model.save_pretrained(out_dir)
    max_pixels = meta.get("max_pixels", DEFAULT_MAX_PIXELS)
    load_processor(base, meta.get("longest_edge"), max_pixels).save_pretrained(out_dir)
    meta.update(base_model=base, lora=False, max_pixels=max_pixels)
    (out_dir / "chashma.json").write_text(json.dumps(meta, indent=2))


def _results_table(reports_dir: Path, aliases: dict[str, str] | None = None) -> str:
    rows = []
    for path in sorted(reports_dir.glob("*.json")):
        r = json.loads(path.read_text())
        model = (aliases or {}).get(r["model"], r["model"])
        dataset = (aliases or {}).get(r["dataset"], r["dataset"])
        rows.append(
            f"| {model} | {dataset} | {r['n']} | {r['accuracy']:.1%} "
            f"| {r['latency_p50_ms']:.0f} ms |"
        )
    if not rows:
        return "_No evaluation reports found._"
    header = "| model | eval set | n | click accuracy | p50 latency |\n|---|---|---|---|---|"
    return "\n".join([header, *rows])


def model_card(
    repo_id: str,
    base_model: str,
    meta: dict,
    reports_dir: Path,
    source_url: str | None = None,
    local_name: str | None = None,
) -> str:
    name = repo_id.split("/")[-1]
    built_by = f" built with [this code]({source_url})" if source_url else ""
    aliases = {
        "chashma-test-n300": "unseen websites (own test set)",
        "screenspot-web": "ScreenSpot web",
    }
    if local_name:
        aliases[local_name] = f"**{name}**"
    trained_on = meta.get("train_samples")
    data_line = (
        f"{trained_on:,} (screenshot, instruction, box) examples"
        if trained_on
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
*"open the pricing page"* and it answers with where to click, as `(x, y)` on a
0..1000 grid relative to the image.

LoRA fine-tune of [{base_model}](https://huggingface.co/{base_model}) on {data_line}{built_by}:
real websites rendered with Playwright, click targets and labels
taken from the DOM, instructions from templates plus a vision LLM. Trained on one free
Kaggle T4; runs in about 5 GB of VRAM.

## Usage

```python
import re

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "{repo_id}"
model = AutoModelForImageTextToText.from_pretrained(repo, dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(repo)

image = Image.open("screenshot.png").convert("RGB")
instruction = "open the pricing page"
prompt = (
    f"Point to the UI element I should click to: {{instruction}}. "
    "Answer with the click point as (x, y)."
)
content = [{{"type": "image"}}, {{"type": "text", "text": prompt}}]
messages = [{{"role": "user", "content": content}}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
reply = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
x, y = (int(v) for v in re.findall(r"\\d+", reply)[:2])
print(x * image.width / 1000, y * image.height / 1000)  # click point in pixels
```

The reply is `(x, y)` on a 0..1000 grid; scale by the image size to get pixels.

## Results

Click accuracy = the predicted point lands inside the target element's box.

{_results_table(reports_dir, aliases)}

## Limitations

- Trained on desktop and mobile *web* pages; native mobile and desktop apps are out of
  distribution.
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
        repo_id,
        meta.get("base_model", DEFAULT_BASE),
        meta,
        reports_dir,
        source_url,
        local_name=model_path.name,
    )
    (out_dir / "README.md").write_text(card)
    if not push:
        return out_dir

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        commit_message="upload model",
        ignore_patterns=["chashma.json"],
    )
    return commit.commit_url
