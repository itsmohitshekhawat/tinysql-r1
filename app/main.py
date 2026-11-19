"""TinySQL-R1 web app (FastAPI): one question, three models side by side.

  /             the demo page
  /api/meta     databases, schemas, example questions, benchmark results, training curve
  /api/ask      runs the question on: the small model untrained, the same model with the GRPO-trained adapter, and
                the frontier model; executes each SQL on the real (read-only) database; streams results as they finish
  /health       for uptime monitoring

The small model runs ON THIS SERVER, on CPU, inside a llama.cpp process (see serve.py). It is loaded once; the
trained LoRA adapter is switched on or off per request, so "before training" and "after training" use the same weights.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tinysql import sqlenv

from . import budget, serve

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = ROOT.parent / "data"

app = FastAPI(title="TinySQL-R1", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
small_lock = threading.Lock()  # the CPU model answers one request at a time


class AskIn(BaseModel):
    db: str
    question: str = Field(..., min_length=3, max_length=300)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def client_ip(request: Request) -> str:
    # Behind a CDN proxy the real visitor address is in the CF-Connecting-IP header.
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


ASSET_VERSION = hashlib.sha256(b"".join((STATIC / f).read_bytes() for f in ("style.css", "app.js"))).hexdigest()[:10]


@app.get("/", include_in_schema=False)
def home() -> HTMLResponse:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    html = html.replace("/static/style.css", f"/static/style.css?v={ASSET_VERSION}")
    html = html.replace("/static/app.js", f"/static/app.js?v={ASSET_VERSION}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health() -> dict:
    return {"ok": True, "small_model": serve.small_ready()}


@app.get("/api/status")
def api_status(request: Request) -> dict:
    return budget.status(client_ip(request))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def gold_for(db: str, question: str) -> dict | None:
    """If the question is one of the test questions, that item (reference SQL + needed columns), for right/wrong marks."""
    for split in ("test_seen", "test_unseen"):
        for item in QUESTIONS.get(split, []):
            if item["db"] == db and item["question"].strip().lower() == question.strip().lower():
                return item
    return None


QUESTIONS = read_json(DATA / "questions.json") or {}


@app.get("/api/meta")
def api_meta() -> JSONResponse:
    schemas = sqlenv.schemas()
    examples = {}
    for split in ("test_unseen", "test_seen"):
        for item in QUESTIONS.get(split, []):
            examples.setdefault(item["db"], [])
            if len(examples[item["db"]]) < 12:
                examples[item["db"]].append({"question": item["question"], "level": item["level"]})
    results = {}
    for name in ("base", "sft", "grpo", "sft_grpo", "deepseek"):
        data = read_json(DATA / "results" / f"{name}.json")
        if data:
            results[name] = {**data["summary"], "select_star": round(
                sum("select *" in r["pred"].lower() for r in data["rows"]) / max(len(data["rows"]), 1), 3)}
    return JSONResponse({
        "databases": {name: {"ddl": s["ddl"], "held_out": s["held_out"], "rows": s["rows"]} for name, s in schemas.items()},
        "examples": examples, "results": results, "training": read_json(DATA / "results" / "training.json"),
        "reward_hacking": read_json(DATA / "results" / "reward_hacking.json"),
        "counts": {k: len(v) for k, v in QUESTIONS.items()}, "small_live": serve.small_ready(),
    })


SAVED = {"base": "base", "trained": "grpo"}  # which results file holds each small-model variant's saved answers


def saved_answer(model: str, db: str, question: str) -> dict | None:
    """The answer recorded during the benchmark, for a test question (used when the small model is not running live)."""
    data = read_json(DATA / "results" / f"{SAVED[model]}.json") or {"rows": []}
    key = question.strip().lower()
    return next((r for r in data["rows"] if r["db"] == db and r["question"].strip().lower() == key), None)


def answer(model: str, db: str, question: str, gold: dict | None) -> dict:
    start = time.perf_counter()
    saved = False
    try:
        if model == "frontier":
            reply, cost = serve.frontier(db, question)
        elif serve.small_ready():
            with small_lock:
                start = time.perf_counter()  # do not count time spent waiting for another visitor
                reply, cost = serve.small(db, question, trained=(model == "trained")), 0.0
        else:
            row = saved_answer(model, db, question)
            if row is None:
                return {"model": model, "offline": True}
            reply, cost, saved = row["pred"], 0.0, True
    except Exception as exc:
        return {"model": model, "failed": type(exc).__name__}
    seconds = round(time.perf_counter() - start, 2)
    sql = sqlenv.extract_sql(reply)
    rows, error = sqlenv.run(db, sql)
    out = {"model": model, "sql": sql, "seconds": seconds, "cost_usd": round(cost, 6), "error": error, "saved": saved,
           "rows": [list(r) for r in (rows or [])[:12]], "row_count": len(rows or [])}
    if gold and error is None:
        out["correct"] = sqlenv.same_result(db, sql, gold["sql"], gold.get("need"))[0]
    elif gold:
        out["correct"] = False
    return out


@app.post("/api/ask")
def api_ask(body: AskIn, request: Request) -> StreamingResponse:
    ip = client_ip(request)

    def send(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    def stream():
        if body.db not in sqlenv.schemas():
            yield send("failed", {"error": "unknown database"})
            return
        state = budget.status(ip)
        if not state["open"]:
            yield send("limit", state)
            return
        gold = gold_for(body.db, body.question)
        yield send("start", {"has_reference": gold is not None})
        results: queue.Queue = queue.Queue()
        models = ["frontier", "base", "trained"]
        for name in models:
            threading.Thread(target=lambda n=name: results.put(answer(n, body.db, body.question.strip(), gold)),
                             daemon=True).start()
        cost = 0.0
        for _ in models:
            try:
                result = results.get(timeout=120)
            except queue.Empty:
                break
            cost += result.get("cost_usd", 0.0)
            yield send("result", result)
        budget.record(ip, cost)
        yield send("done", {})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
