from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from chashma.data.build import load_samples
from chashma.data.schema import GroundingSample
from chashma.model.format import build_prompt, encode_point
from chashma.model.grounder import chat_messages, load_processor

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
    num_workers: int = 2
    seed: int = 0
    report_to: str = "none"
    lora: LoraConfig = field(default_factory=LoraConfig)

    @classmethod
    def from_yaml(cls, path: Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(path.read_text()) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        lora = LoraConfig(**raw.pop("lora", {}))
        for key in ("dataset", "output_dir"):
            if key in raw:
                raw[key] = Path(raw[key])
        return cls(**raw, lora=lora)


class GroundingDataset:
    def __init__(self, samples: list[GroundingSample], image_root: Path):
        self.samples = samples
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        return {
            "image": Image.open(self.image_root / s.image).convert("RGB"),
            "prompt": build_prompt(s.instruction),
            "answer": encode_point(s.box.center),
        }


class Collator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "right"
        image_token = getattr(processor, "image_token", "<image>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)

    def _answer_len(self, prompt_text: str, full_text: str) -> int:
        n_full = len(self.tokenizer(full_text, add_special_tokens=False).input_ids)
        n_prompt = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
        return n_full - n_prompt

    def __call__(self, batch: list[dict]) -> dict:
        prompts, fulls = [], []
        for ex in batch:
            prompts.append(
                self.processor.apply_chat_template(
                    chat_messages(ex["prompt"]), add_generation_prompt=True
                )
            )
            fulls.append(
                self.processor.apply_chat_template(chat_messages(ex["prompt"], ex["answer"]))
            )
        enc = self.processor(
            text=fulls,
            images=[[ex["image"]] for ex in batch],
            return_tensors="pt",
            padding=True,
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, (p, f) in enumerate(zip(prompts, fulls, strict=True)):
            seq_len = int(enc["attention_mask"][i].sum())
            labels[i, : seq_len - self._answer_len(p, f)] = -100
        labels[labels == self.image_token_id] = -100
        enc["labels"] = labels
        return enc


def _subset(samples: list, n: int | None, seed: int) -> list:
    if n is None or n >= len(samples):
        return samples
    return random.Random(seed).sample(samples, n)


def train(cfg: TrainConfig) -> Path:
    import torch
    from transformers import AutoModelForImageTextToText, Trainer, TrainingArguments

    train_samples, image_root = load_samples(cfg.dataset, "train")
    val_samples, _ = load_samples(cfg.dataset, "val")
    train_samples = _subset(train_samples, cfg.max_train_samples, cfg.seed)
    val_samples = _subset(val_samples, cfg.max_eval_samples, cfg.seed)
    log.info("train=%d val=%d", len(train_samples), len(val_samples))

    processor = load_processor(cfg.base_model, cfg.longest_edge)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.base_model, dtype=torch.bfloat16, attn_implementation="sdpa"
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
        bf16=True,
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
        train_dataset=GroundingDataset(train_samples, image_root),
        eval_dataset=GroundingDataset(val_samples, image_root) if has_val else None,
        data_collator=Collator(processor),
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    processor.save_pretrained(str(cfg.output_dir))
    meta = {
        "base_model": cfg.base_model,
        "longest_edge": cfg.longest_edge,
        "epochs": cfg.epochs,
        "train_samples": len(train_samples),
        "lora": cfg.lora.enabled,
    }
    (cfg.output_dir / "chashma.json").write_text(json.dumps(meta, indent=2))
    return cfg.output_dir
