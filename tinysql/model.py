"""Load the small model (optionally with a trained LoRA adapter) and generate SQL."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import sqlenv

BASE = "Qwen/Qwen2.5-0.5B-Instruct"  # 494M parameters, Apache-2.0
MAX_NEW_TOKENS = 96


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(adapter: str | None = None, trainable: bool = False):
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"  # decoder-only models must be padded on the left for batched generation
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(device())
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if not trainable:
        model.eval()
    return tok, model


def prompt(tok, db: str, question: str) -> str:
    return tok.apply_chat_template(sqlenv.messages(db, question), tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompts: list[str], sample: bool = False, temperature: float = 1.0) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=sample, temperature=temperature if sample else None,
                         top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
