"""Small training and checkpoint interfaces shared by all experiments."""

import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import GPT2Config
from .model import GPT2Model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_token_loss(model, input_ids, attention_mask, labels):
    hidden = model(input_ids, attention_mask)["last_hidden_state"][:, :-1]
    targets = labels[:, 1:]
    valid = (targets != -100) & attention_mask[:, 1:].bool()
    if not valid.any():
        raise ValueError("batch has no supervised next-token targets")
    # project only supervised positions; nli needs one target per example.
    logits = model.hidden_state_to_token(hidden[valid])
    return torch.nn.functional.cross_entropy(logits.float(), targets[valid])


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    accumulation_steps=1,
    mixed_precision=False,
    log_interval=50,
):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    total = len(loader)
    if total == 0:
        raise ValueError("training loader is empty")
    use_amp = mixed_precision and str(device).startswith("cuda")
    if use_amp and not torch.cuda.is_bf16_supported():
        raise ValueError("mixed precision requires bfloat16 GPU support")
    for step, batch in enumerate(loader):
        tensors = {
            key: batch[key].to(device)
            for key in ("input_ids", "attention_mask", "labels")
        }
        group_start = (step // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total - group_start)
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else nullcontext()
        )
        with context:
            loss = next_token_loss(model, **tensors)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at batch {step}")
        (loss / group_size).backward()
        losses.append(loss.item())
        if (step + 1) % accumulation_steps == 0 or step + 1 == total:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if log_interval and (step + 1) % log_interval == 0:
            print(
                json.dumps(
                    {"batch": step + 1, "batches": total, "loss": losses[-1]}
                ),
                flush=True,
            )
    return losses


def save_checkpoint(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1:
        raise ValueError("expected a version 1 checkpoint with configuration")
    model = GPT2Model(GPT2Config(**bundle["config"]))
    model.load_state_dict(bundle["state_dict"])
    return model.to(device).eval(), bundle["metadata"]
