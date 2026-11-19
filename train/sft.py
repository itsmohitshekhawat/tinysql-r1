"""Supervised fine-tuning (SFT): the classic comparison for GRPO.

SFT shows the model the reference SQL and trains it to copy it, token by token (cross-entropy loss).
GRPO never sees reference SQL; it only gets a score for its own attempts.
Comparing the two on the same data answers: "how much does learning from a reward cost compared with learning from
labelled answers?" It also gives a good starting point for GRPO (--init runs/sft).

Usage: python train/sft.py --epochs 2 --out runs/sft
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from common import ROOT, accuracy, lora, m, pad_batch, splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()

    torch.manual_seed(opts.seed)
    rng = random.Random(opts.seed)
    train, dev = splits()
    tok, net = m.load()
    net = lora(net)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=opts.lr)
    core = net.get_base_model()
    body, head = core.model, core.lm_head
    out = ROOT / opts.out
    out.mkdir(parents=True, exist_ok=True)
    log = {"config": vars(opts), "steps": [], "dev": [{"step": 0, "accuracy": accuracy(tok, net, dev)}]}
    print(f"step 0  dev accuracy {log['dev'][0]['accuracy']:.3f}", flush=True)
    best, step, start = log["dev"][0]["accuracy"], 0, time.time()

    for epoch in range(opts.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), opts.batch):
            chunk = train[i:i + opts.batch]
            prompts = [tok(m.prompt(tok, x["db"], x["question"]))["input_ids"] for x in chunk]
            answers = [tok(x["sql"])["input_ids"] + [tok.eos_token_id] for x in chunk]
            ids, mask = pad_batch(tok, [p + a for p, a in zip(prompts, answers)], net.device)
            hidden = body(input_ids=ids, attention_mask=mask).last_hidden_state
            loss = 0.0
            for row, (p, a) in enumerate(zip(prompts, answers)):  # loss only on the answer tokens
                lo, hi = len(p) - 1, len(p) + len(a) - 1
                loss = loss + torch.nn.functional.cross_entropy(head(hidden[row, lo:hi]).float(), ids[row, lo + 1:hi + 1])
            loss = loss / len(chunk)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            step += 1
            log["steps"].append({"step": step, "loss": round(float(loss.detach()), 4)})
            if step % 10 == 0:
                print(f"epoch {epoch + 1} step {step}  loss {float(loss.detach()):.3f}  {(time.time() - start) / step:.1f}s/step", flush=True)
        acc = accuracy(tok, net, dev)
        log["dev"].append({"step": step, "accuracy": acc})
        print(f"epoch {epoch + 1} done  dev accuracy {acc:.3f}", flush=True)
        if acc >= best:
            best = acc
            net.save_pretrained(out)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"done in {(time.time() - start) / 60:.1f} min, best dev accuracy {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
