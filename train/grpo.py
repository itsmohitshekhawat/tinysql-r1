"""GRPO (Group Relative Policy Optimization) for text-to-SQL, written from scratch in PyTorch.

The idea (from the DeepSeekMath / DeepSeek-R1 papers), in plain words:
  1. Take a question. Let the model write G different answers by sampling (a "group").
  2. Score every answer with a reward that can be CHECKED: run the SQL, compare the rows (tinysql/sqlenv.py).
  3. Compare each answer with its own group: advantage = (reward - group mean) / group std.
     Better-than-average answers get a positive advantage, worse ones a negative one. No separate "critic"
     network is needed, which is what makes GRPO simple and cheap.
  4. Nudge the model to make the tokens of good answers more likely and the tokens of bad answers less likely:
     loss = - mean( advantage * log-probability of the sampled tokens ).

No reference SQL is shown to the model. It only ever learns from "did your query return the right rows?".

Simplifications compared with the paper (kept honest on purpose):
- One gradient step per sampled batch. The new and old policy are then identical at update time, so PPO's
  probability ratio is 1 and its clipping never activates; the objective reduces to the loss above.
- No KL penalty to the starting model (beta = 0). LoRA, a small learning rate and gradient clipping keep it stable.

Usage: python train/grpo.py --steps 300 --out runs/grpo [--init runs/sft]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from common import ROOT, accuracy, lora, m, pad_batch, splits, sqlenv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--questions", type=int, default=4, help="questions per step")
    ap.add_argument("--group", type=int, default=8, help="sampled answers per question (G)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=4, help="sequences per backward pass (memory limit)")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--init", help="start from an existing LoRA adapter (e.g. the supervised one)")
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()

    torch.manual_seed(opts.seed)
    rng = random.Random(opts.seed)
    train, dev = splits()
    tok, net = m.load(opts.init, trainable=True) if opts.init else m.load()
    if not opts.init:
        net = lora(net)
    net.train()
    params = [p for p in net.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=opts.lr, weight_decay=0.0)
    out = ROOT / opts.out
    out.mkdir(parents=True, exist_ok=True)
    log = {"config": vars(opts), "trainable_params": sum(p.numel() for p in params), "steps": [], "dev": []}
    best = accuracy(tok, net, dev)
    log["dev"].append({"step": 0, "accuracy": best})
    print(f"step 0  dev accuracy {best:.3f}  trainable params {log['trainable_params']:,}", flush=True)
    eos = tok.eos_token_id
    core = net.get_base_model()  # the model with the LoRA layers inside
    body, head = core.model, core.lm_head
    start = time.time()

    for step in range(1, opts.steps + 1):
        batch = rng.sample(train, opts.questions)
        prompts = [m.prompt(tok, x["db"], x["question"]) for x in batch for _ in range(opts.group)]

        # 1. Sample a group of answers per question (no gradients needed while sampling).
        net.eval()
        with torch.no_grad():
            enc = tok(prompts, return_tensors="pt", padding=True).to(net.device)
            gen = net.generate(**enc, max_new_tokens=m.MAX_NEW_TOKENS, do_sample=True, temperature=opts.temperature,
                               top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        net.train()
        completions = gen[:, enc["input_ids"].shape[1]:].tolist()
        answers = []
        for ids in completions:  # keep tokens up to and including the first end-of-sequence token
            answers.append(ids[: ids.index(eos) + 1] if eos in ids else ids)

        # 2. Verifiable reward for every answer.
        rewards = torch.tensor([sqlenv.reward(batch[i // opts.group]["db"], tok.decode(a, skip_special_tokens=True),
                                              batch[i // opts.group]["sql"], batch[i // opts.group].get("need"))
                                for i, a in enumerate(answers)])

        # 3. Advantage inside each group.
        grouped = rewards.view(opts.questions, opts.group)
        adv = (grouped - grouped.mean(dim=1, keepdim=True)) / (grouped.std(dim=1, keepdim=True) + 1e-4)
        adv = adv.view(-1)
        useful = [i for i in range(len(answers)) if adv[i].abs() > 1e-6 and len(answers[i]) > 0]
        # A group where every answer got the same reward teaches nothing (advantage 0), so it is skipped.

        # 4. Policy-gradient step, in small chunks so it fits in laptop memory.
        optim.zero_grad(set_to_none=True)
        loss_value = 0.0
        prompt_ids = [tok(p)["input_ids"] for p in prompts]
        for c in range(0, len(useful), opts.chunk):
            idx = useful[c:c + opts.chunk]
            seqs = [prompt_ids[i] + answers[i] for i in idx]
            ids, mask = pad_batch(tok, seqs, net.device)
            # Memory: the vocabulary has 151,000 tokens, so scores for every position of every sequence would need
            # gigabytes. We run the transformer body once, then compute vocabulary scores ONLY at the answer positions.
            hidden = body(input_ids=ids, attention_mask=mask).last_hidden_state
            loss = 0.0
            for row, i in enumerate(idx):
                a, b = len(prompt_ids[i]) - 1, len(prompt_ids[i]) + len(answers[i]) - 1  # positions that predict the answer tokens
                logits = head(hidden[row, a:b]).float()
                logp = -torch.nn.functional.cross_entropy(logits, ids[row, a + 1:b + 1], reduction="mean")
                loss = loss - adv[i].to(net.device) * logp
            loss = loss / max(len(useful), 1)
            loss.backward()
            loss_value += float(loss.detach())
            del hidden, loss
        if useful:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
        # Apple's GPU backend keeps freed memory blocks cached; with changing tensor shapes the process grew to
        # 10 GB and started swapping. Dropping references and emptying the cache each step keeps it flat.
        del enc, gen
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        entry = {"step": step, "reward": round(float(rewards.mean()), 4), "correct": round(float((rewards == 1.0).float().mean()), 4),
                 "valid_sql": round(float((rewards > 0).float().mean()), 4), "used": len(useful), "loss": round(loss_value, 4)}
        log["steps"].append(entry)
        if step % 5 == 0:
            recent = log["steps"][-25:]
            print(f"step {step:4d}  reward(25) {sum(e['reward'] for e in recent) / len(recent):.3f}  "
                  f"correct(25) {sum(e['correct'] for e in recent) / len(recent):.3f}  used {len(useful):2d}/{len(answers)}  "
                  f"{(time.time() - start) / step:.1f}s/step", flush=True)
        if step % opts.eval_every == 0 or step == opts.steps:
            acc = accuracy(tok, net, dev)
            log["dev"].append({"step": step, "accuracy": acc})
            print(f"step {step:4d}  dev accuracy {acc:.3f} (best {best:.3f})", flush=True)
            if acc >= best:  # keep the best adapter seen so far
                best = acc
                net.save_pretrained(out)
            (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"done in {(time.time() - start) / 60:.1f} min, best dev accuracy {best:.3f}, adapter in {out}", flush=True)


if __name__ == "__main__":
    main()
