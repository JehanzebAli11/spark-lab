"""Smoke test for the vLLM teacher: three Roman Urdu prompts against the OpenAI-compatible API.

    python3 -m lab teacher-smoke [--base-url http://localhost:8000/v1] [--model google/gemma-4-31B-it]

Checks that the teacher answers each prompt, in Latin script (Roman Urdu, not Nastaliq) and with
common Urdu function words. The answers are printed and never saved: this checks the server,
it does not make training data.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://localhost:8000/v1"

SYSTEM = (
    "Hamesha Roman Urdu mein jawab do, yani Urdu zabaan magar English huroof mein. "
    "Urdu rasm-ul-khat ya English mein jawab na do. Jawab chhota rakho."
)

PROMPTS = [
    "Do jumlon mein batao ke doodh patti chai kaise banti hai.",
    "Apne dost ke liye Eid Mubarak ka ek chhota sa paighaam likho.",
    "Kya yeh message scam lagta hai? 'Aap ka account band ho raha hai, foran is link par apna PIN bhejein.' "
    "Haan ya nahi, aur ek wajah batao.",
]

# Common Urdu function words as written in Roman Urdu. Two distinct hits are enough evidence.
ROMAN_URDU_MARKERS = frozenset(
    {
        "hai", "hain", "ka", "ki", "ke", "ko", "mein", "main", "aur", "se", "ye", "yeh", "woh", "nahi",
        "nahin", "kya", "aap", "ap", "hum", "tum", "ho", "kar", "karo", "karein", "bhi", "par", "pe", "jo",
        "ek", "liye", "hota", "hoti", "tha", "thi", "raha", "rahi", "ga", "gi", "lagta", "chahiye",
    }
)

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z']+")


@dataclass
class Result:
    prompt: str
    answer: str
    ok: bool
    reason: str
    seconds: float


def check_roman_urdu(text: str) -> tuple[bool, str]:
    """Return (ok, reason) for whether `text` looks like a Roman Urdu answer."""
    text = text.strip()
    if not text:
        return False, "empty answer"
    arabic, latin = len(_ARABIC.findall(text)), len(_LATIN.findall(text))
    if arabic > 0.2 * max(1, arabic + latin):
        return False, "answer is in Urdu script, not Roman Urdu"
    words = {w.lower().strip("'") for w in _WORD.findall(text)}
    hits = sorted(words & ROMAN_URDU_MARKERS)
    if len(hits) < 2:
        return False, "no Roman Urdu function words (English answer?)"
    return True, "Roman Urdu (" + ", ".join(hits[:5]) + ")"


def _opener(base_url: str) -> urllib.request.OpenerDirector:
    host = urllib.parse.urlparse(base_url).hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        # Never send local traffic through an HTTP(S)_PROXY.
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _call(base_url: str, path: str, body: dict | None, timeout: float) -> dict:
    url = base_url.rstrip("/") + path
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    req.add_header("Content-Type", "application/json")
    with _opener(base_url).open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(base_url: str, timeout: float = 10) -> list[str]:
    return [m["id"] for m in _call(base_url, "/models", None, timeout).get("data", [])]


def ask(base_url: str, model: str, prompt: str, timeout: float = 300) -> str:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    out = _call(base_url, "/chat/completions", body, timeout)
    return (out["choices"][0]["message"].get("content") or "").strip()


def run(base_url: str, model: str | None, timeout: float = 300, out=None) -> int:
    """Run the three prompts. Returns 0 if all pass, 1 if any fail, 2 if the server is unreachable."""
    out = out or sys.stdout
    try:
        models = list_models(base_url)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"teacher not reachable at {base_url}: {exc}", file=out)
        return 2
    if not model:
        if not models:
            print("teacher lists no models", file=out)
            return 2
        model = models[0]
    elif model not in models:
        print(f"teacher serves {models}, not {model}", file=out)
        return 2

    results: list[Result] = []
    for i, prompt in enumerate(PROMPTS, 1):
        start = time.monotonic()
        try:
            answer = ask(base_url, model, prompt, timeout)
            ok, reason = check_roman_urdu(answer)
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            answer, ok, reason = "", False, f"request failed: {exc}"
        r = Result(prompt, answer, ok, reason, time.monotonic() - start)
        results.append(r)
        print(f"[{i}/{len(PROMPTS)}] {'PASS' if ok else 'FAIL'} ({r.seconds:.1f}s, {reason})", file=out)
        print(f"  Q: {prompt}", file=out)
        for line in (answer or "-").splitlines():
            print(f"  A: {line}", file=out)
    passed = sum(r.ok for r in results)
    print(f"teacher smoke: {passed}/{len(results)} passed ({model})", file=out)
    return 0 if passed == len(results) else 1


def main(base_url: str | None = None, model: str | None = None, timeout: float = 300) -> int:
    base_url = base_url or os.environ.get("TEACHER_URL") or DEFAULT_BASE_URL
    model = model or os.environ.get("TEACHER_MODEL") or None
    return run(base_url, model, timeout)
