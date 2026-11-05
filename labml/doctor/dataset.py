"""Build the training-run doctor dataset: runs, references, curve images.

    python -m labml.doctor.dataset --per-cause 450 --procs 4

For every cause and seed it trains the injected run and a healthy reference on the same
task rows (`sim.reference_config`). A run is **harmed** when the bad setting actually hurt:

  train_test_leak   reported val accuracy beats clean test accuracy by > 3 points
  label_noise       always (train and val labels are wrong)
  healthy           never, by definition (we still record how it compares)
  overfitting       as below, or clean test loss > 1.5 x reference + 0.05 (over-confidence;
                    the same loss rule would flag 13% of healthy runs, so this label is noisier)
  everything else   clean test accuracy > 2 points below the reference, or the loss diverged

Splits: seed % 10 in 0-6 train, 7 val, 8-9 test, all on the four standard task families;
`heldout_task` runs use the `rings` task, which no model sees in training.

Writes:
  datasets/doctor-runs/runs.jsonl.gz     config, logged series, final metrics, reference, harmed, split
  datasets/doctor-runs/images.npz        every run's curve image, 64x64 RGB uint8, in runs.jsonl order
  data/doctor/images/<id>.png            the same images at 160x120 (not committed; big)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .sim import CAUSES, HELDOUT_TASKS, TASKS, reference_config, run, sample_config  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "datasets" / "doctor-runs"
IMG_DIR = REPO / "data" / "doctor" / "images"


def split_of(task: str, seed: int) -> str:
    if task in HELDOUT_TASKS:
        return "heldout_task"
    r = seed % 10
    return "train" if r < 7 else "val" if r == 7 else "test"


def harmed(cause: str, res: dict, ref: dict) -> bool:
    f, rf = res["final"], ref["final"]
    if cause == "healthy":
        return False
    if cause == "label_noise":
        return True
    if cause == "train_test_leak":
        return bool(f["val_acc"] - f["test_acc"] > 0.03)
    if cause == "overfitting" and f["test_loss"] is not None and math.isfinite(f["test_loss"]) \
            and f["test_loss"] > 1.5 * rf["test_loss"] + 0.05:
        return True  # over-confident: accuracy often survives overfitting, test loss does not
    return bool(f["diverged"] or not math.isfinite(f["test_acc"]) or f["test_acc"] < rf["test_acc"] - 0.02)


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def one(args: tuple[str, str, int]) -> dict:
    cause, task, seed = args
    cfg = sample_config(cause, task, seed)
    t = time.time()
    res = run(cfg)
    ref = run(reference_config(cfg))
    return _clean({"id": f"{cause}-{task}-{seed}", "cause": cause, "task": task, "seed": seed,
                   "split": split_of(task, seed), "config": res["config"], "n_params": res["n_params"],
                   "series": res["series"], "final": res["final"], "reference": ref["final"],
                   "harmed": harmed(cause, res, ref), "seconds": round(time.time() - t, 2)})


def tasks(per_cause: int, heldout_per_cause: int) -> list[tuple[str, str, int]]:
    out = []
    for cause in CAUSES:
        out += [(cause, TASKS[s % len(TASKS)], s) for s in range(per_cause)]
        out += [(cause, HELDOUT_TASKS[0], 10_000 + s) for s in range(heldout_per_cause)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labml.doctor.dataset")
    ap.add_argument("--per-cause", type=int, default=450)
    ap.add_argument("--heldout-per-cause", type=int, default=40)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)

    todo = tasks(a.per_cause, a.heldout_per_cause)
    # longest first so the pool finishes evenly
    todo.sort(key=lambda t: t[0] in ("overfitting", "train_test_leak", "no_warmup", "label_noise"), reverse=True)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(one, todo, chunksize=2), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"{i}/{len(todo)} runs, {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (CAUSES.index(r["cause"]), r["task"], r["seed"]))
    with gzip.open(a.out / "runs.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} runs in {time.time() - t0:.0f} s")

    from .render import render_all

    imgs = render_all(rows, IMG_DIR, procs=a.procs)
    np.savez_compressed(a.out / "images.npz", images=imgs, ids=np.array([r["id"] for r in rows]))
    print(f"images: {imgs.shape} -> {a.out / 'images.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
