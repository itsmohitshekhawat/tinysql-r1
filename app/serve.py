"""Model calls for the demo.

Small model: Qwen2.5-0.5B-Instruct as a GGUF file, served by a local llama.cpp server process on CPU. The LoRA
adapter trained with GRPO is loaded next to it; each request sets its scale to 0 (untrained) or 1 (trained).
Frontier model: DeepSeek through an OpenAI-compatible API.
"""

from __future__ import annotations

import json
import os
import urllib.request

from openai import OpenAI

from tinysql import sqlenv

SMALL_URL = os.environ.get("SMALL_URL", "http://127.0.0.1:8081")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash")
PRICE_IN, PRICE_OUT = 0.15, 0.60  # USD per 1M tokens, used only if the API does not report the cost

_client: OpenAI | None = None


def small_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{SMALL_URL}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def small(db: str, question: str, trained: bool) -> str:
    body = {"messages": sqlenv.messages(db, question), "max_tokens": 96, "temperature": 0,
            "lora": [{"id": 0, "scale": 1.0 if trained else 0.0}], "cache_prompt": True}
    req = urllib.request.Request(f"{SMALL_URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def frontier(db: str, question: str) -> tuple[str, float]:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"), timeout=60, max_retries=1)
    r = _client.chat.completions.create(model=MODEL, messages=sqlenv.messages(db, question), max_tokens=600, temperature=0,
                                        extra_body={"usage": {"include": True}})
    cost = getattr(r.usage, "cost", None)
    if cost is None:
        cost = (r.usage.prompt_tokens * PRICE_IN + r.usage.completion_tokens * PRICE_OUT) / 1e6
    return r.choices[0].message.content or "", float(cost)
