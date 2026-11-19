"""Cost control for a public demo: a daily USD budget for the whole site + a daily turn limit per visitor.

Stored in a small SQLite file. Visitors are counted by a hash of their IP (the raw IP is never stored).
When the budget is used up, the site still works: the results page and the benchmark numbers are free.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path

DAILY_BUDGET_USD = float(os.environ.get("DAILY_BUDGET_USD", "0.50"))
PER_VISITOR_PER_DAY = int(os.environ.get("PER_VISITOR_PER_DAY", "30"))
DB_PATH = Path(os.environ.get("BUDGET_DB", "/tmp/tinysql-budget.sqlite"))

_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS usage (day TEXT, visitor TEXT, questions INTEGER, cost REAL,"
                " PRIMARY KEY (day, visitor))")
    return con


def today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def visitor_id(ip: str) -> str:
    return hashlib.sha256(f"tinysql:{ip}".encode()).hexdigest()[:16]


def status(ip: str) -> dict:
    with _lock, _db() as con:
        spent = con.execute("SELECT COALESCE(SUM(cost), 0) FROM usage WHERE day = ?", (today(),)).fetchone()[0]
        asked = con.execute("SELECT COALESCE(questions, 0) FROM usage WHERE day = ? AND visitor = ?",
                            (today(), visitor_id(ip))).fetchone()
    asked = asked[0] if asked else 0
    return {"spent_today_usd": round(spent, 4), "budget_usd": DAILY_BUDGET_USD,
            "your_questions_left": max(0, PER_VISITOR_PER_DAY - asked),
            "open": spent < DAILY_BUDGET_USD and asked < PER_VISITOR_PER_DAY}


def record(ip: str, cost: float) -> None:
    with _lock, _db() as con:
        con.execute("INSERT INTO usage VALUES (?, ?, 1, ?) ON CONFLICT(day, visitor) DO UPDATE SET"
                    " questions = questions + 1, cost = cost + excluded.cost", (today(), visitor_id(ip), cost))


