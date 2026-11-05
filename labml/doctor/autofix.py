"""Auto-fix loop: diagnose, change the setting, rerun, repeat (up to 3 rounds).

    python -m labml.doctor.autofix

For every harmed run in test + heldout_task (and every healthy one, to check that the
doctor leaves good runs alone) it compares four policies:

  none      rerun unchanged with a new seed (does luck fix it?)
  doctor    features doctor; stop when it says healthy or is below its confidence threshold
  ensemble  image + features doctor, same rule
  oracle    apply the fix for the true cause (upper bound for the fixes themselves)

A run is **rescued** when the final rerun is no longer harmed, with the same rule the
dataset uses (dataset.harmed) against the same healthy reference.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

from .dataset import _clean, harmed  # noqa: E402
from .features import ALL, matrix  # noqa: E402
from .models import CNN, MODELS, RESULTS, load, usable  # noqa: E402
from .render import png_bytes, to_array  # noqa: E402
from .sim import CAUSES, RunConfig, apply_fix, run  # noqa: E402

ROUNDS = 3
_STATE: dict = {}


def _init() -> None:
    import torch

    torch.set_num_threads(1)
    with open(MODELS / "doctors.pkl", "rb") as fh:
        _STATE.update(pickle.load(fh))
    cnn = CNN(len(CAUSES))
    cnn.net.load_state_dict(torch.load(MODELS / "cnn.pt"))
    cnn.classes_ = _STATE["classes"]
    _STATE["cnn"] = cnn


def diagnose(row: dict, doctor: str) -> tuple[str, float]:
    classes = _STATE["classes"]
    p = _STATE["features"].predict_proba(matrix([row], ALL))[0]
    if doctor == "ensemble":
        img = to_array(png_bytes(row["series"]))[None]
        p = (p + _STATE["cnn"].predict_proba(img)[0]) / 2
    return str(classes[int(np.argmax(p))]), float(np.max(p))


def _as_row(res: dict, base: dict) -> dict:
    return _clean({**base, "config": res["config"], "series": res["series"], "final": res["final"],
                   "n_params": res["n_params"]})


def loop(row: dict, policy: str) -> dict:
    cfg = RunConfig(**row["config"])
    cause, ref = row["cause"], {"final": row["reference"]}
    current, steps = row, []
    if policy == "none":
        res = run(replace(cfg, seed=cfg.seed + 1))
        current = _as_row(res, row)
        steps.append({"diagnosis": "-", "fix": "rerun with new seed"})
    else:
        threshold = _STATE["thresholds"]["ensemble" if policy == "ensemble" else "features"]
        for _ in range(ROUNDS):
            if policy == "oracle":
                diag, conf = (cause if harmed(cause, current, ref) else "healthy"), 1.0
            else:
                diag, conf = diagnose(current, policy)
            if diag == "healthy" or conf < threshold:
                steps.append({"diagnosis": diag, "confidence": round(conf, 3),
                              "fix": "stop" if diag == "healthy" else "escalate"})
                break
            cfg = apply_fix(cfg, diag)
            steps.append({"diagnosis": diag, "confidence": round(conf, 3), "fix": diag})
            current = _as_row(run(cfg), row)
    still = harmed(cause, current, ref) if cause != "healthy" else (
        current["final"]["test_acc"] is None or current["final"]["test_acc"] < row["final"]["test_acc"] - 0.02)
    return {"id": row["id"], "cause": cause, "split": row["split"], "policy": policy, "steps": steps,
            "reruns": sum(1 for s in steps if s["fix"] not in ("stop", "escalate")),
            "before": row["final"], "after": current["final"], "reference": row["reference"],
            "rescued": (not still) if cause != "healthy" else None,
            "hurt_healthy": still if cause == "healthy" else None}


def _job(args):
    row, policy = args
    return loop(row, policy)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labml.doctor.autofix")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-per-cause", type=int, default=0, help="cap runs per cause and split (0 = all)")
    a = ap.parse_args(argv)
    rows, _ = load()
    ok = usable(rows)
    pick = [r for r, u in zip(rows, ok) if u and r["split"] in ("test", "heldout_task")]
    if a.max_per_cause:
        seen: Counter = Counter()
        kept = []
        for r in pick:
            seen[(r["cause"], r["split"])] += 1
            if seen[(r["cause"], r["split"])] <= a.max_per_cause:
                kept.append(r)
        pick = kept
    jobs = [(r, p) for r in pick for p in ("none", "doctor", "ensemble", "oracle")]
    t0 = time.time()
    out = []
    with Pool(a.procs, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, jobs, chunksize=2), 1):
            out.append(res)
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} loops, {time.time() - t0:.0f} s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "autofix_runs.jsonl", "w", encoding="utf-8") as fh:
        for r in sorted(out, key=lambda r: (r["id"], r["policy"])):
            fh.write(json.dumps(r) + "\n")
    summary = summarise(out)
    (RESULTS / "autofix.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS / "autofix.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    return 0


def summarise(out: list[dict]) -> dict:
    s: dict = {"by_policy": {}, "by_cause": defaultdict(dict), "healthy": {}}
    for policy in ("none", "doctor", "ensemble", "oracle"):
        broken = [r for r in out if r["policy"] == policy and r["cause"] != "healthy"]
        healthy = [r for r in out if r["policy"] == policy and r["cause"] == "healthy"]
        s["by_policy"][policy] = {
            "harmed_runs": len(broken), "rescued": sum(r["rescued"] for r in broken),
            "rescue_rate": round(sum(r["rescued"] for r in broken) / max(1, len(broken)), 4),
            "reruns": sum(r["reruns"] for r in broken),
            "escalated": sum(1 for r in broken if r["steps"] and r["steps"][-1]["fix"] == "escalate"),
            "heldout_task_rescue_rate": round(
                sum(r["rescued"] for r in broken if r["split"] == "heldout_task")
                / max(1, sum(1 for r in broken if r["split"] == "heldout_task")), 4)}
        s["healthy"][policy] = {"n": len(healthy), "touched": sum(1 for r in healthy if r["reruns"]),
                                "hurt": sum(bool(r["hurt_healthy"]) for r in healthy)}
        for c in CAUSES[1:]:
            rs = [r for r in broken if r["cause"] == c]
            s["by_cause"][c][policy] = f"{sum(r['rescued'] for r in rs)}/{len(rs)}"
    s["by_cause"] = dict(s["by_cause"])
    return s


def markdown(s: dict) -> str:
    out = ["# Auto-fix loop", "",
           "Harmed runs from test + held-out task. Up to 3 rounds of diagnose -> change setting -> rerun. "
           "Rescued = the final rerun is no longer harmed against the same healthy reference.", "",
           "| Policy | Harmed runs | Rescued | Rescue rate | Held-out task rescue rate | Reruns | Escalated "
           "| Healthy runs touched | Healthy runs hurt |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p, v in s["by_policy"].items():
        h = s["healthy"][p]
        out.append(f"| {p} | {v['harmed_runs']} | {v['rescued']} | {v['rescue_rate']:.1%} | "
                   f"{v['heldout_task_rescue_rate']:.1%} | {v['reruns']} | {v['escalated']} | "
                   f"{h['touched']}/{h['n']} | {h['hurt']}/{h['n']} |")
    out += ["", "Rescued per cause:", "", "| Cause | none | doctor | ensemble | oracle |",
            "| --- | --- | --- | --- | --- |"]
    for c, v in s["by_cause"].items():
        out.append(f"| {c} | {v['none']} | {v['doctor']} | {v['ensemble']} | {v['oracle']} |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
