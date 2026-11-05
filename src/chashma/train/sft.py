from __future__ import annotations

import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from chashma.data.build import load_samples
from chashma.data.schema import GroundingSample
from chashma.model.format import build_prompt, encode_point
from chashma.model.grounder import load_processor

log = logging.getLogger(__name__)


@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    base_model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    dataset: Path = Path("data/dataset")
    image_root: Path | None = None
    output_dir: Path = Path("checkpoints/chashma-256m")
    longest_edge: int = 1536
    epochs: float = 2.0
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True
    freeze_vision: bool = False
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    max_train_samples: int | None = None
    max_eval_samples: int | None = 300
    pack_turns: int = 16
    num_workers: int = 2
    seed: int = 0
    report_to: str = "none"
    resume: bool = False
    keep_checkpoints: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)

    @classmethod
    def from_yaml(cls, path: Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(path.read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        lora = LoraConfig(**raw.pop("lora", {}))
        for key in ("dataset", "output_dir", "image_root"):
            if key in raw:
                raw[key] = Path(raw[key])
        return cls(**raw, lora=lora)


def pack_by_image(
    samples: list[GroundingSample], max_turns: int, seed: int = 0
) -> list[list[GroundingSample]]:
    by_image: dict[str, list[GroundingSample]] = defaultdict(list)
    for s in samples:
        by_image[s.image].append(s)
    rng = random.Random(seed)
    groups = []
    for key in sorted(by_image):
        by_element: dict[tuple, list[GroundingSample]] = defaultdict(list)
        for s in by_image[key]:
            by_element[tuple(s.bbox)].append(s)
        queues = list(by_element.values())
        for q in queues:
            rng.shuffle(q)
        rng.shuffle(queues)
        ordered = [q[k] for k in range(max(map(len, queues))) for q in queues if k < len(q)]
        group: list[GroundingSample] = []
        for s in ordered:
            if len(group) >= max_turns or any(g.bbox == s.bbox for g in group):
                groups.append(group)
                group = []
            group.append(s)
        groups.append(group)
    return groups


def conversation(turns: list[tuple[str, str]]) -> list[dict]:
    messages = []
    for i, (prompt, answer) in enumerate(turns):
        content = [{"type": "image"}] if i == 0 else []
        messages.append({"role": "user", "content": [*content, {"type": "text", "text": prompt}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


class GroundingDataset:
    def __init__(self, groups: list[list[GroundingSample]], image_root: Path):
        self.groups = groups
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        group = self.groups[i]
        return {
            "image": Image.open(self.image_root / group[0].image).convert("RGB"),
            "turns": [(build_prompt(s.instruction), encode_point(s.box.center)) for s in group],
        }


class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "right"

    def _ntok(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _render(self, turns: list[tuple[str, str]]) -> tuple[str, list[tuple[int, int]]]:
        template = self.processor.apply_chat_template
        messages = conversation(turns)
        full = template(messages)
        spans = []
        for k in range(len(turns)):
            prompt = template(messages[: 2 * k + 1], add_generation_prompt=True)
            done = template(messages[: 2 * k + 2])
            if not (full.startswith(done) and done.startswith(prompt)):
                raise ValueError("chat template is not prefix-stable, can't locate answers")
            spans.append((self._ntok(prompt), self._ntok(done)))
        return full, spans

    def __call__(self, batch: list[dict]) -> dict:
        rendered = [self._render(ex["turns"]) for ex in batch]
        enc = self.processor(
            text=[full for full, _ in rendered],
            images=[[ex["image"]] for ex in batch],
            return_tensors="pt",
            padding=True,
        )
        ids = enc["input_ids"]
        labels = ids.new_full(ids.shape, -100)
        for i, (full, spans) in enumerate(rendered):
            # <image> becomes hundreds of tokens, which shifts the answer spans
            shift = int(enc["attention_mask"][i].sum()) - self._ntok(full)
            for start, end in spans:
                labels[i, start + shift : end + shift] = ids[i, start + shift : end + shift]
        enc["labels"] = labels
        return enc


def _subset(samples: list, n: int | None, seed: int) -> list:
    if n is None or n >= len(samples):
        return samples
    return random.Random(seed).sample(samples, n)


def train(cfg: TrainConfig) -> Path:
    import torch
    from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    train_samples, image_root = load_samples(cfg.dataset, "train", cfg.image_root)
    val_samples, _ = load_samples(cfg.dataset, "val", cfg.image_root)
    train_samples = _subset(train_samples, cfg.max_train_samples, cfg.seed)
    val_samples = _subset(val_samples, cfg.max_eval_samples, cfg.seed)
    train_groups = pack_by_image(train_samples, cfg.pack_turns, cfg.seed)
    val_groups = pack_by_image(val_samples, cfg.pack_turns, cfg.seed)
    log.info(
        "train=%d samples in %d sequences, val=%d samples",
        len(train_samples), len(train_groups), len(val_samples),
    )

    processor = load_processor(cfg.base_model, cfg.longest_edge)
    # no bf16 on t4, so fp32 weights + fp16 autocast
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.base_model,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        attn_implementation="sdpa",
    )
    if cfg.freeze_vision:
        for name, param in model.named_parameters():
            if "vision_model" in name:
                param.requires_grad = False
    if cfg.lora.enabled:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        model = get_peft_model(
            model,
            PeftLoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
                task_type="CAUSAL_LM",
            ),
        )
        if cfg.gradient_checkpointing:
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    has_val = len(val_samples) > 0
    args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        lr_scheduler_type="cosine",
        warmup_steps=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=bf16,
        fp16=cuda and not bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.num_workers,
        seed=cfg.seed,
        report_to=cfg.report_to,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=GroundingDataset(train_groups, image_root),
        eval_dataset=GroundingDataset(val_groups, image_root) if has_val else None,
        data_collator=Collator(processor),
    )
    last = get_last_checkpoint(str(cfg.output_dir)) if cfg.output_dir.is_dir() else None
    if cfg.resume and last:
        log.info("resuming from %s", last)
    trainer.train(resume_from_checkpoint=last if cfg.resume else None)
    trainer.model.to(torch.bfloat16)
    trainer.save_model(str(cfg.output_dir))
    processor.save_pretrained(str(cfg.output_dir))
    meta = {
        "base_model": cfg.base_model,
        "longest_edge": cfg.longest_edge,
        "epochs": cfg.epochs,
        "train_samples": len(train_samples),
        "pack_turns": cfg.pack_turns,
        "lora": cfg.lora.enabled,
    }
    (cfg.output_dir / "chashma.json").write_text(json.dumps(meta, indent=2))
    if not cfg.keep_checkpoints:
        for ckpt in cfg.output_dir.glob("checkpoint-*"):
            shutil.rmtree(ckpt)
    return cfg.output_dir
