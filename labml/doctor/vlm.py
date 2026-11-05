"""VLM doctor: ask a vision-language model on the Spark to diagnose a run from its curve picture.

    python -m labml.doctor.vlm --url http://localhost:8000/v1 --model google/gemma-4-31b-it --limit 300

Talks to any OpenAI-compatible chat endpoint (vLLM on the Spark). Standard library only.
For each test / held-out-task run it renders the 320x240 curve picture (same drawing as the
image doctor), sends it with the legend and the list of causes, and parses a JSON answer.
Writes results/doctor/vlm.json and vlm.md. Evaluating a model is allowed by CLAUDE.md;
its answers are not training data unless the owner decides otherwise (Gemma on the Spark is
the allowed teacher).

Not run in the cloud session (no GPU, no model); tested against a fake server only.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.request
from pathlib import Path

CAUSES = ("healthy", "lr_too_high", "lr_too_low", "no_warmup", "label_noise", "overfitting",
          "train_test_leak", "dead_activations", "exploding_gradients")

PROMPT = """You are diagnosing a small neural-network training run from its training curves.

The picture: blue solid line = training loss, orange solid line = validation loss (both on a
shared log-scale y axis, steps on x). Green dashed line = validation accuracy on a 0-1 axis
on the right. A red X marks where the loss became NaN or infinite and training stopped.

Choose the single most likely cause:
- healthy: nothing wrong
- lr_too_high: learning rate far too high
- lr_too_low: learning rate far too low
- no_warmup: a deep net started at full learning rate with no warmup
- label_noise: many training and validation labels are wrong
- overfitting: tiny training set, model memorises it
- train_test_leak: validation set contains copies of training rows
- dead_activations: most ReLU units output zero
- exploding_gradients: very deep net with too-large initialisation

Answer with JSON only: {"cause": "<one of the names above>", "confidence": <0 to 1>, "reason": "<short>"}"""


def ask(url: str, model: str, png: bytes, timeout: float = 120.0) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": 200,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            ]}]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        text = json.loads(r.read())["choices"][0]["message"]["content"]
    return parse(text)


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    cause = str(d.get("cause", "")).strip()
    if cause not in CAUSES:
        cause = next((c for c in CAUSES if c in text), "unparsed")
    try:
        conf = float(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"cause": cause, "confidence": max(0.0, min(1.0, conf)), "reason": str(d.get("reason", ""))[:300],
            "raw": text[:500]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labml.doctor.vlm")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--limit", type=int, default=300, help="runs to ask about (spread over causes)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "doctor")
    a = ap.parse_args(argv)

    from .models import load, usable
    from .render import png_bytes

    rows, _ = load()
    pick = [r for r, u in zip(rows, usable(rows)) if u and r["split"] in ("test", "heldout_task")]
    per = max(1, a.limit // len(CAUSES))
    chosen = []
    for c in CAUSES:
        chosen += [r for r in pick if r["cause"] == c][:per]
    res = evaluate_rows(chosen, a.url, a.model, lambda s: png_bytes(s, 320, 240))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "vlm.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (a.out / "vlm.md").write_text(markdown(res["summary"]), encoding="utf-8")
    print(markdown(res["summary"]))
    return 0


def evaluate_rows(rows: list[dict], url: str, model: str, render) -> dict:
    """Ask about each row; `pred` is the VLM's cause, `cause` the injected one."""
    answers = []
    for r in rows:
        t = time.time()
        try:
            a = ask(url, model, render(r["series"]))
        except OSError as exc:
            a = {"cause": "error", "confidence": 0.0, "reason": str(exc)[:200], "raw": ""}
        answers.append({"id": r["id"], "cause": r["cause"], "split": r["split"], "pred": a["cause"],
                        "confidence": a["confidence"], "reason": a["reason"], "raw": a["raw"],
                        "seconds": round(time.time() - t, 2)})
    summary: dict = {"n": len(answers), "model": model, "by_split": {}, "per_cause": {}}
    for s in sorted({o["split"] for o in answers}):
        os_ = [o for o in answers if o["split"] == s]
        summary["by_split"][s] = {"n": len(os_), "accuracy": round(sum(o["cause"] == o["pred"] for o in os_)
                                                                   / len(os_), 4)}
    for c in CAUSES:
        os_ = [o for o in answers if o["cause"] == c]
        if os_:
            summary["per_cause"][c] = {"n": len(os_), "recall": round(sum(o["pred"] == c for o in os_) / len(os_), 4)}
    summary["unparsed"] = sum(o["pred"] in ("unparsed", "error") for o in answers)
    return {"summary": summary, "answers": answers}


def markdown(s: dict) -> str:
    out = [f"# VLM doctor ({s['model']})", "", f"{s['n']} runs, {s['unparsed']} unparsed or failed answers.", "",
           "| Split | n | Accuracy |", "| --- | --- | --- |"]
    out += [f"| {k} | {v['n']} | {v['accuracy']:.3f} |" for k, v in s["by_split"].items()]
    out += ["", "| Cause | n | Recall |", "| --- | --- | --- |"]
    out += [f"| {k} | {v['n']} | {v['recall']:.3f} |" for k, v in s["per_cause"].items()]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
