"""Measure execution accuracy: the share of questions where the model's SQL returns exactly the right rows.

  python scripts/evaluate.py base                         the untrained small model
  python scripts/evaluate.py grpo --adapter runs/grpo     a trained LoRA adapter
  python scripts/evaluate.py deepseek --api               the frontier model over its API (needs LLM_API_KEY)

Writes data/results/<name>.json with every question, the model's SQL, whether it was right, and the time taken.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tinysql import sqlenv  # noqa: E402

SPLITS = ("test_seen", "test_unseen")


def score(item: dict, reply: str) -> dict:
    sql = sqlenv.extract_sql(reply)
    ok, err = sqlenv.same_result(item["db"], sql, item["sql"], item.get("need"))
    return {**item, "pred": sql, "correct": ok, "error": err}


def local(name: str, adapter: str | None, items: list[dict], batch: int) -> tuple[list[dict], float]:
    from tinysql import model as m

    tok, net = m.load(adapter)
    rows, start = [], time.perf_counter()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        replies = m.generate(tok, net, [m.prompt(tok, x["db"], x["question"]) for x in chunk])
        rows += [score(x, r) for x, r in zip(chunk, replies)]
    return rows, (time.perf_counter() - start) / len(items)


def api(items: list[dict]) -> tuple[list[dict], float, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"), timeout=60)
    model = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash")

    def one(x):
        t = time.perf_counter()
        r = client.chat.completions.create(model=model, messages=sqlenv.messages(x["db"], x["question"]), max_tokens=600,
                                           temperature=0, extra_body={"usage": {"include": True}})
        row = score(x, r.choices[0].message.content or "")
        row["seconds"] = round(time.perf_counter() - t, 2)
        row["cost"] = float(getattr(r.usage, "cost", 0) or 0)
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, items))
    return rows, sum(r["seconds"] for r in rows) / len(rows), sum(r["cost"] for r in rows) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--adapter")
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    opts = ap.parse_args()
    questions = json.loads((ROOT / "data" / "questions.json").read_text())
    items = [{**x, "split": s} for s in SPLITS for x in questions[s]]
    if opts.limit:
        items = items[:opts.limit]
    cost = 0.0
    if opts.api:
        rows, seconds, cost = api(items)
    else:
        rows, seconds = local(opts.name, opts.adapter, items, opts.batch)
    summary = {"name": opts.name, "seconds_per_question": round(seconds, 3), "cost_per_question_usd": round(cost, 6)}
    for split in SPLITS + ("all",):
        part = [r for r in rows if split == "all" or r["split"] == split]
        summary[split] = round(sum(r["correct"] for r in part) / max(len(part), 1), 3)
        for level in ("easy", "medium", "hard"):
            lv = [r for r in part if r["level"] == level]
            summary[f"{split}_{level}"] = round(sum(r["correct"] for r in lv) / max(len(lv), 1), 3)
    summary["runs_without_error"] = round(sum(r["error"] is None for r in rows) / len(rows), 3)
    out = ROOT / "data" / "results"
    out.mkdir(exist_ok=True)
    (out / f"{opts.name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
