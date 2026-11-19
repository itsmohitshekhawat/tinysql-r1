"""Build the question sets. A teacher model (DeepSeek) writes questions with reference SQL for each database; every
pair is then VERIFIED by running the SQL: it must run, be a SELECT, and return between 1 and 50 rows. Pairs that fail
are thrown away, so wrong teacher answers never reach the data.

Splits:
  train        questions on the 6 training databases
  test_seen    different questions on the same 6 databases
  test_unseen  questions on 2 databases the model never sees in training (movies, sports)

Usage: LLM_API_KEY=... python scripts/make_questions.py   -> data/questions.json  (about $0.05)
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tinysql import sqlenv  # noqa: E402

PER_TRAIN_DB, PER_HELD_OUT_DB, BATCH = 130, 80, 26
TEST_SEEN_SHARE = 0.15
LEVELS = """- easy: one table, a filter, a sort or a count
- medium: one join, or GROUP BY with an aggregate, or a LIMIT with ORDER BY
- hard: two joins, HAVING, a subquery, or several conditions together"""

client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"), timeout=120)
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash")


def ask(db: str, batch_no: int) -> list[dict]:
    ddl = sqlenv.schemas()[db]["ddl"]
    sample = {t: sqlenv.run(db, f"SELECT * FROM {t} LIMIT 3")[0] for t in re.findall(r"CREATE TABLE (\w+)", ddl)}
    prompt = f"""Write {BATCH} different natural-language questions about this SQLite database, each with the SQL query that answers it.

Schema:
{ddl}

Sample rows: {json.dumps(sample, default=str)}

Rules:
- Mix of difficulty, about one third each:
{LEVELS}
- Questions must sound like a real person asking, and must be answerable from the schema alone.
- Use values that exist in the sample rows or are plausible (cities, categories, years).
- SQLite syntax only. One SELECT statement per question. No parameters.
- This is batch {batch_no}: vary topics, tables and phrasing; avoid the most obvious questions.

Return only a JSON list: [{{"question": "...", "sql": "...", "level": "easy|medium|hard"}}, ...]"""
    resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=6000,
                                          temperature=0.9)
    text = resp.choices[0].message.content or ""
    m = re.search(r"\[.*\]", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []


def valid(db: str, item: dict) -> bool:
    sql = str(item.get("sql", "")).strip().rstrip(";")
    if not item.get("question") or ";" in sql:
        return False
    rows, err = sqlenv.run(db, sql)
    return err is None and 1 <= len(rows) <= 50


def main() -> None:
    rng = random.Random(7)
    jobs = [(db, b) for db, s in sqlenv.schemas().items()
            for b in range((PER_HELD_OUT_DB if s["held_out"] else PER_TRAIN_DB) // BATCH)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(lambda j: (j[0], ask(*j)), jobs))
    out = {"train": [], "test_seen": [], "test_unseen": []}
    stats = {}
    for db in sqlenv.schemas():
        raw = [i for d, items in batches if d == db for i in items if isinstance(i, dict)]
        seen, kept = set(), []
        for item in raw:
            key = re.sub(r"\W+", " ", str(item.get("question", "")).lower()).strip()
            if key in seen or not valid(db, item):
                continue
            seen.add(key)
            kept.append({"db": db, "question": item["question"].strip(), "sql": item["sql"].strip().rstrip(";"),
                         "level": item.get("level", "medium")})
        rng.shuffle(kept)
        stats[db] = f"{len(kept)}/{len(raw)}"
        if sqlenv.schemas()[db]["held_out"]:
            out["test_unseen"] += kept
        else:
            cut = int(len(kept) * TEST_SEEN_SHARE)
            out["test_seen"] += kept[:cut]
            out["train"] += kept[cut:]
    (ROOT / "data" / "questions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("kept/generated per database:", stats)
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
