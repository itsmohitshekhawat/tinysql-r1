"""Shared training helpers: LoRA setup, the dev split and a quick accuracy check."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tinysql import model as m  # noqa: E402
from tinysql import sqlenv  # noqa: E402

DEV_SIZE = 48


def lora(net, r: int = 16):
    """LoRA: freeze the model and train small extra matrices in the attention and MLP layers (about 1-2% of the
    weights). Cheap enough for a laptop, and the adapter file is a few MB."""
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(net, cfg)


def splits(seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Training questions, minus a small dev set used to watch progress (the test sets stay untouched)."""
    items = json.loads((ROOT / "data" / "questions.json").read_text())["train"]
    random.Random(seed).shuffle(items)
    return items[DEV_SIZE:], items[:DEV_SIZE]


@torch.no_grad()
def accuracy(tok, net, items: list[dict], batch: int = 16) -> float:
    net.eval()
    right = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        replies = m.generate(tok, net, [m.prompt(tok, x["db"], x["question"]) for x in chunk])
        right += sum(sqlenv.same_result(x["db"], sqlenv.extract_sql(r), x["sql"], x.get("need"))[0] for x, r in zip(chunk, replies))
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    net.train()
    return right / len(items)


def pad_batch(tok, sequences: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token id lists into a batch (for the training forward pass)."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, s in enumerate(sequences):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)
