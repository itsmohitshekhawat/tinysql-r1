"""Label, for every question, which columns of the reference result the question really asks for.

Why: reference queries often return helper columns ("Which doctors earned more than 500?" -> name, total_fees).
A check that demands every reference column punishes correct short answers; a check that accepts any subset can be
gamed (and was: see README, "reward hacking"). So each question gets a field "need": the indices of the reference
columns an answer must contain. The teacher model proposes them; plain code validates the indices.

  python scripts/label_columns.py        adds "need" to data/questions.json (needs LLM_API_KEY)
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tinysql import sqlenv  # noqa: E402

RULES = """For each item you get a question, the reference SQL and the numbered columns that SQL returns.
Decide which columns an answer MUST contain to answer the question as asked.

Rules:
- A column the question asks to see is needed ("show the name and salary", "how many", "the average price", "list titles").
- "Which X ..." / "Who ..." / "Find the X ..." needs the column that identifies X (usually its name or title).
- A number the question asks for is needed: "how many orders each customer placed" needs the customer AND the count.
- A column that is only there to help (the value used for filtering or sorting, an extra id, a total the question
  did not ask to see) is NOT needed.
- At least one column is always needed. If unsure, mark the column as needed.

Reply with JSON only: {"need": [[indices for item 1], [indices for item 2], ...]}, one list per item, in order."""


def label(client: OpenAI, model: str, chunk: list[dict]) -> list[list[int]]:
    lines = []
    for n, x in enumerate(chunk, 1):
        cols = ", ".join(f"{i}: {c}" for i, c in enumerate(x["_cols"]))
        lines.append(f"Item {n}\nQuestion: {x['question']}\nSQL: {x['sql']}\nColumns: {cols}")
    for _ in range(3):
        r = client.chat.completions.create(model=model, temperature=0, max_tokens=1500, response_format={"type": "json_object"},
                                           messages=[{"role": "system", "content": RULES}, {"role": "user", "content": "\n\n".join(lines)}])
        try:
            need = json.loads(r.choices[0].message.content)["need"]
            if len(need) == len(chunk):
                return need
        except (ValueError, KeyError, TypeError):
            pass
    return [[] for _ in chunk]  # falls back to "all columns" below


def main() -> None:
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"), timeout=120)
    model = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash")
    path = ROOT / "data" / "questions.json"
    data = json.loads(path.read_text())
    items = [x for part in data.values() for x in part]
    for x in items:
        x["_cols"] = sqlenv.columns(x["db"], x["sql"])
    chunks = [items[i:i + 12] for i in range(0, len(items), 12)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda c: label(client, model, c), chunks))
    fallback = 0
    for chunk, needs in zip(chunks, results):
        for x, need in zip(chunk, needs):
            width = len(x.pop("_cols"))
            ok = isinstance(need, list) and need and all(isinstance(i, int) and 0 <= i < width for i in need)
            fallback += not ok
            x["need"] = sorted(set(need)) if ok else list(range(width))  # invalid label -> require every column
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    trimmed = sum(len(x["need"]) < len(sqlenv.columns(x["db"], x["sql"])) for x in items)
    print(f"{len(items)} questions labelled; {trimmed} have optional helper columns; {fallback} fell back to all columns")


if __name__ == "__main__":
    main()
