"""The SQL environment: build the prompt, run a query safely, compare results, and score an answer (the reward).

This is what makes the task "verifiable": we never need a human or another AI to judge an answer. We run the model's
SQL on the real database and compare the rows with the rows of the reference query.
"""

from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DBS = ROOT / "data" / "dbs"
SYSTEM = "You translate questions into SQLite SQL. Reply with one SQL query and nothing else."
MAX_STEPS = 200_000  # SQLite VM steps before a query is cancelled (stops runaway joins)
MAX_ROWS = 200


@lru_cache(maxsize=1)
def schemas() -> dict:
    return json.loads((ROOT / "data" / "schemas.json").read_text())


def messages(db: str, question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Database schema:\n{schemas()[db]['ddl']}\n\nQuestion: {question}"}]


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply: strip code fences, reasoning tags and anything after the first statement."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\b(SELECT|WITH)\b.*", text, flags=re.S | re.I)
    sql = (m.group(0) if m else text).strip()
    return sql.split(";")[0].strip()


def run(db: str, sql: str) -> tuple[list[tuple] | None, str | None]:
    """Run a read-only query. Returns (rows, None) or (None, error message)."""
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.I):
        return None, "only SELECT queries are allowed"
    con = sqlite3.connect(f"file:{DBS / (db + '.sqlite')}?mode=ro", uri=True)
    steps = 0

    def tick() -> int:
        nonlocal steps
        steps += 1000
        return 1 if steps > MAX_STEPS else 0  # non-zero cancels the query

    con.set_progress_handler(tick, 1000)
    try:
        rows = con.execute(sql).fetchmany(MAX_ROWS + 1)
        return [tuple(r) for r in rows[:MAX_ROWS]], None
    except sqlite3.Error as exc:
        return None, str(exc)[:200]
    finally:
        con.close()


def _cell(v):
    return round(v, 2) if isinstance(v, float) else v


def columns(db: str, sql: str) -> list[str]:
    """Names of the columns a query returns (used when labelling which reference columns a question asks for)."""
    con = sqlite3.connect(f"file:{DBS / (db + '.sqlite')}?mode=ro", uri=True)
    try:
        return [d[0] for d in con.execute(sql).description]
    finally:
        con.close()


def rows_match(pred: list[tuple], gold: list[tuple], need: list[int] | None = None) -> bool:
    """Execution match on the columns the question asks for.

    `need` lists the reference columns the question really asks for (data/questions.json, field "need"; None = all).
    The reference query often returns helper columns too: "Which doctors earned more than 500?" has the reference
    (name, total_fees), and a query returning only the names is right. But "show each city and its number of
    customers" needs both columns.

    A result matches when it has the same number of rows and every needed column can be found among its columns,
    giving the same rows (as a multiset: row order is ignored). Extra columns are allowed; missing ones are not.

    The first version of this check accepted ANY subset of the reference columns. Reinforcement learning found that
    hole within 50 steps and learned to return one easy column instead of answering (see README, "reward hacking").
    """
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    need = list(range(len(gold[0]))) if need is None else need
    if len(pred[0]) < len(need):
        return False
    want = [tuple(_cell(r[k]) for k in need) for r in gold]
    got = [tuple(_cell(v) for v in r) for r in pred]
    target = sorted(want, key=repr)
    # For each needed column, the columns of the model's result that hold the same values.
    options = []
    for c in range(len(need)):
        column = sorted((r[c] for r in want), key=repr)
        options.append([k for k in range(len(got[0])) if sorted((r[k] for r in got), key=repr) == column])

    def search(c: int, chosen: tuple[int, ...]) -> bool:
        if c == len(need):
            return sorted((tuple(r[k] for k in chosen) for r in got), key=repr) == target
        return any(search(c + 1, chosen + (k,)) for k in options[c] if k not in chosen)

    return search(0, ())


def same_result(db: str, sql: str, gold_sql: str, need: list[int] | None = None) -> tuple[bool, str | None]:
    """Does the query return the rows of the reference query, on the columns the question asks for?"""
    rows, err = run(db, sql)
    if err is not None:
        return False, err
    gold, gold_err = run(db, gold_sql)
    if gold_err is not None:
        raise ValueError(f"reference query failed: {gold_err}")
    return rows_match(rows, gold, need), None


def reward(db: str, reply: str, gold_sql: str, need: list[int] | None = None) -> float:
    """The training signal for reinforcement learning.
      1.0  the query returns the right rows, with every column the question asks for
      0.2  the query runs but returns the wrong rows (valid SQL is a step in the right direction)
      0.0  not a query, or it fails to run
    """
    sql = extract_sql(reply)
    ok, err = same_result(db, sql, gold_sql, need)
    if ok:
        return 1.0
    return 0.2 if err is None else 0.0
