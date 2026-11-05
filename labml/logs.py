"""Chaos log dataset and the CPU log classifier.

    python -m labml.logs build  --seeds 80 --out datasets/chaos-logs/logs.jsonl.gz
    python -m labml.logs train  --data datasets/chaos-logs/logs.jsonl.gz

`build` runs every chaos fault (lab/chaos.py) through the real queue (`Queue.run`, healing
off, no retries) in throwaway lab roots and keeps the tail the healer would see. The label
is the fault that was injected, so it is exact.

Splits (by fault variant and seed, fixed in code):
  train / val / test   in-distribution variants, seeds split 70% / 15% / 15%
  heldout_val          one unseen variant per class, used only to pick the threshold
  heldout_test         another unseen variant per class, reported, never tuned on

`train` fits TF-IDF + logistic regression with scikit-learn, compares it with the regex
rules, picks the confidence threshold on val + heldout_val, and exports weights to
lab/models/log_classifier.json for the standard-library classifier in lab/triage.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from lab.chaos import FAULTS, STYLES
from lab.paths import lab_paths
from lab.queue import Queue
from lab.triage import CLASSES, UNKNOWN, LinearClassifier, RuleClassifier, error_region, tail, tokens

REPO = Path(__file__).resolve().parent.parent
HELDOUT_VAL = ("oom.numpy_alloc", "disk_full.quota", "cuda_error.xid", "network.dns", "bad_data.csv_float",
               "flaky.port", "code_bug.zerodiv")
HELDOUT_TEST = ("oom.vllm_kv", "disk_full.sqlite", "cuda_error.cudnn", "network.reset", "bad_data.pickle",
                "flaky.eintr", "code_bug.unbound")
_TMP = re.compile(r"/tmp/[\w.-]+")


def split_of(fault: str, seed: int, n_seeds: int) -> str:
    if fault in HELDOUT_VAL:
        return "heldout_val"
    if fault in HELDOUT_TEST:
        return "heldout_test"
    r = seed / n_seeds
    return "train" if r < 0.7 else "val" if r < 0.85 else "test"


def scrub(text: str, root: str) -> str:
    """Remove machine-specific paths (CLAUDE.md rule 7)."""
    text = text.replace(root, "/home/lab/lab")
    text = text.replace(str(REPO), "/home/lab/lab/spark-lab")
    text = text.replace(os.path.expanduser("~"), "/home/lab")
    return _TMP.sub("/tmp/job", text)


def _one(args: tuple[str, int, int]) -> dict:
    fault, seed, n_seeds = args
    rng = random.Random(f"{fault}:{seed}")
    style = rng.choice(STYLES)
    with tempfile.TemporaryDirectory(prefix="chaos-") as root:
        paths = lab_paths(root).ensure()
        q = Queue(paths, notifier=lambda *a, **k: True)
        name = f"chaos-{fault.replace('.', '-')}-{seed}"
        run_dir = paths.runs / "chaos" / name
        for step in range(50, 50 * rng.randint(3, 6) + 1, 50):
            (run_dir / f"checkpoint-{step}").mkdir(parents=True)
        job = {"name": name, "cmd": [sys.executable, "-m", "lab.chaos", "--fault", fault, "--seed", str(seed),
                                     "--style", style],
               "env": {"PYTHONPATH": str(REPO), "BATCH_SIZE": str(rng.choice([8, 16, 32, 64, 128]))},
               "timeout_min": 2, "retries": 0, "heal": False, "notify": False}
        if rng.random() < 0.3:
            job["resume_dir"] = f"runs/chaos/{name}"
        q.enqueue(job)
        done = q.run(q.claim())
        text = (paths.root / done.log).read_text(encoding="utf-8", errors="replace")
        return {"id": f"{fault}:{seed}", "fault": fault, "label": FAULTS[fault].label, "real": FAULTS[fault].real,
                "style": style, "seed": seed, "split": split_of(fault, seed, n_seeds), "returncode": done.returncode,
                "text": scrub(tail(text), root)}


def build(seeds: int, out: Path, procs: int) -> list[dict]:
    tasks = [(f, s, seeds) for f in sorted(FAULTS) for s in range(seeds)]
    with Pool(procs) as pool:
        rows = pool.map(_one, tasks, chunksize=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out}")
    print(Counter((r["split"], r["label"]) for r in rows))
    return rows


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


# ---- training and evaluation -------------------------------------------------------------


def _metrics(gold: list[str], pred: list[str]) -> dict:
    """Accuracy over all rows, plus coverage and precision when `unknown` means escalate."""
    n = len(gold)
    accepted = [(g, p) for g, p in zip(gold, pred) if p != UNKNOWN]
    correct = sum(g == p for g, p in accepted)
    per_class = {}
    for c in CLASSES:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0, "support": tp + fn}
    return {"n": n, "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / n, 4),
            "coverage": round(len(accepted) / n, 4),
            "precision_when_acting": round(correct / len(accepted), 4) if accepted else 0.0,
            "wrong_actions": len(accepted) - correct, "escalated": n - len(accepted),
            "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(CLASSES), 4), "per_class": per_class}


def _with_threshold(probs: list[dict[str, float]], t: float) -> list[str]:
    out = []
    for p in probs:
        label = max(p, key=p.get)
        out.append(label if p[label] >= t else UNKNOWN)
    return out


def export(vec, clf, threshold: float, meta: dict) -> dict:
    vocab = {t: int(i) for t, i in vec.vocabulary_.items()}
    return {"classes": [str(c) for c in clf.classes_], "vocab": vocab,
            "idf": [round(float(x), 6) for x in vec.idf_],
            "coef": [[round(float(w), 5) for w in row] for row in clf.coef_],
            "intercept": [round(float(b), 5) for b in clf.intercept_], "threshold": threshold, "meta": meta}


def train(data: Path, model_out: Path, results: Path) -> dict:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rows = load(data)
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test", "heldout_val", "heldout_test")}
    results.mkdir(parents=True, exist_ok=True)

    def fit(train_rows, c):
        vec = TfidfVectorizer(analyzer=tokens, sublinear_tf=True, min_df=2, max_features=5000)
        x = vec.fit_transform([r["text"] for r in train_rows])
        clf = LogisticRegression(C=c, max_iter=3000)
        clf.fit(x, [r["label"] for r in train_rows])
        return vec, clf

    def probs(vec, clf, rs):
        p = clf.predict_proba(vec.transform([r["text"] for r in rs]))
        return [dict(zip(clf.classes_, map(float, row))) for row in p]

    # 1. pick C by leave-one-variant-out CV on train (plus heldout_val): rewards generalising
    #    to fault phrasings the model has not seen, not memorising the ones it has.
    variants = sorted({r["fault"] for r in by["train"]})
    folds = [variants[i::5] for i in range(5)]
    sweep = {}
    for c in (0.1, 0.3, 1, 3, 10, 30):
        gold, pred = [], []
        for fold in folds:
            tr = [r for r in by["train"] if r["fault"] not in fold]
            te = [r for r in by["train"] if r["fault"] in fold]
            vec, clf = fit(tr, c)
            gold += [r["label"] for r in te]
            pred += _with_threshold(probs(vec, clf, te), 0.0)
        vec, clf = fit(by["train"], c)
        gold += [r["label"] for r in by["heldout_val"]]
        pred += _with_threshold(probs(vec, clf, by["heldout_val"]), 0.0)
        sweep[c] = _metrics(gold, pred)["macro_f1"]
        print(f"C={c}: unseen-variant macro-F1 {sweep[c]:.3f}", flush=True)
    best_c = max(sweep, key=lambda c: (sweep[c], -c))
    vec, clf = fit(by["train"], best_c)

    # 2. threshold: the lowest one whose precision-when-acting on the tuning set is >= 0.98
    tune = by["val"] + by["heldout_val"]
    tune_p = probs(vec, clf, tune)
    gold_t = [r["label"] for r in tune]
    curve = []
    for t in np.round(np.arange(0.2, 0.96, 0.025), 3):
        m = _metrics(gold_t, _with_threshold(tune_p, float(t)))
        curve.append({"threshold": float(t), "coverage": m["coverage"], "precision": m["precision_when_acting"]})
    ok = [c for c in curve if c["precision"] >= 0.98]
    threshold = ok[0]["threshold"] if ok else max(curve, key=lambda c: c["precision"])["threshold"]

    # 3. evaluate rules, model (no threshold) and model + threshold on every split
    rules = RuleClassifier()

    def hybrid(rs, p, t):
        """A rule match in the error region wins; otherwise the model, escalating below t."""
        out = []
        for r, m in zip(rs, _with_threshold(p, t)):
            rl = rules.predict(error_region(r["text"])).label
            out.append(rl if rl != UNKNOWN else m)
        return out

    report = {"C_sweep_macro_f1": sweep, "C": best_c, "threshold": threshold, "threshold_curve": curve,
              "n_features": len(vec.vocabulary_), "splits": {}}
    for s, rs in by.items():
        gold = [r["label"] for r in rs]
        p = probs(vec, clf, rs)
        report["splits"][s] = {
            "rules": _metrics(gold, [rules.predict(r["text"]).label for r in rs]),
            "model": _metrics(gold, _with_threshold(p, 0.0)),
            "model+threshold": _metrics(gold, _with_threshold(p, threshold)),
            "hybrid+threshold": _metrics(gold, hybrid(rs, p, threshold)),
        }
    # serving mode: whichever of model / hybrid makes fewer wrong actions on the tuning splits
    def wrong(key):
        return sum(report["splits"][s][key]["wrong_actions"] for s in ("val", "heldout_val"))

    def esc(key):
        return sum(report["splits"][s][key]["escalated"] for s in ("val", "heldout_val"))

    mode = min(("model", "hybrid"), key=lambda m: (wrong(f"{m}+threshold"), esc(f"{m}+threshold")))
    report["mode"] = mode
    meta = {"trained_on": str(data), "n_train": len(by["train"]), "C": best_c, "mode": mode,
            "heldout_test_macro_f1": report["splits"]["heldout_test"]["model"]["macro_f1"]}
    spec = export(vec, clf, threshold, meta)
    spec["mode"] = mode

    # 4. the exported stdlib model must agree with scikit-learn
    lin = LinearClassifier(spec)
    check = by["heldout_test"] + by["test"]
    sk = probs(vec, clf, check)
    diff = max(abs(lin.probabilities(r["text"])[k] - p[k]) for r, p in zip(check, sk) for k in p)
    report["export_max_prob_diff"] = diff
    assert diff < 1e-3, f"stdlib export disagrees with sklearn by {diff}"

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    (results / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _confusions(rows, by, vec, clf, threshold, results)
    _misses(by, vec, clf, threshold, results)
    (results / "report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return report


def _confusions(rows, by, vec, clf, threshold, results: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(CLASSES) + [UNKNOWN]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, split in zip(axes, ("test", "heldout_test")):
        rs = by[split]
        p = clf.predict_proba(vec.transform([r["text"] for r in rs]))
        pred = _with_threshold([dict(zip(clf.classes_, row)) for row in p], threshold)
        m = [[sum(1 for r, q in zip(rs, pred) if r["label"] == g and q == c) for c in labels] for g in CLASSES]
        ax.imshow(m, cmap="Blues")
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                if v:
                    ax.text(j, i, v, ha="center", va="center", fontsize=8,
                            color="white" if v > max(map(max, m)) / 2 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("predicted (unknown = escalated)")
        ax.set_ylabel("injected fault")
        ax.set_title(f"{split} (n={len(rs)}, threshold {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(results / "confusion.png", dpi=120)
    plt.close(fig)


def _misses(by, vec, clf, threshold, results: Path) -> None:
    lines = ["# Held-out-variant mistakes and escalations", ""]
    rs = by["heldout_test"]
    p = clf.predict_proba(vec.transform([r["text"] for r in rs]))
    seen = Counter()
    for r, row in zip(rs, p):
        d = dict(zip(clf.classes_, row))
        lab = max(d, key=d.get)
        if lab == r["label"] and d[lab] >= threshold:
            continue
        key = (r["fault"], lab if d[lab] >= threshold else UNKNOWN)
        seen[key] += 1
        if seen[key] > 1:
            continue
        last = [ln for ln in r["text"].splitlines() if ln.strip()][-3:]
        lines += [f"## {r['fault']} -> {key[1]} ({d[lab]:.2f} {lab})", "", "```", *last, "```", ""]
    lines += ["## Counts", ""] + [f"- {f} -> {p_}: {n}" for (f, p_), n in sorted(seen.items())]
    (results / "heldout_misses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown(rep: dict) -> str:
    out = ["# Log classifier results", "",
           f"TF-IDF ({rep['n_features']} features) + logistic regression, C={rep['C']} (leave-one-variant-out CV), "
           f"confidence threshold {rep['threshold']:.3f} (lowest with >= 98% precision on val + heldout_val).",
           "Rows below the threshold are escalated (incident file) instead of auto-fixed.", "",
           "| Split | n | Rules acc | Model acc | Model macro-F1 | +threshold: acts on | precision when acting "
           "| wrong actions | hybrid: acts on | hybrid precision | hybrid wrong |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s, m in rep["splits"].items():
        t, h = m["model+threshold"], m["hybrid+threshold"]
        out.append(f"| {s} | {m['model']['n']} | {m['rules']['accuracy']:.3f} | {m['model']['accuracy']:.3f} | "
                   f"{m['model']['macro_f1']:.3f} | {t['coverage']:.1%} | {t['precision_when_acting']:.3f} | "
                   f"{t['wrong_actions']} | {h['coverage']:.1%} | {h['precision_when_acting']:.3f} | "
                   f"{h['wrong_actions']} |")
    out += ["", f"Serving mode picked on val + heldout_val: **{rep['mode']}**."]
    out += ["", "Rules accuracy counts a rule miss (`unknown`) as wrong. `heldout_test` uses fault variants "
            "the model never saw, one per class.", "",
            "Per-class F1 on heldout_test (model, no threshold):", "",
            "| Class | Precision | Recall | F1 | n |", "| --- | --- | --- | --- | --- |"]
    for c, v in rep["splits"]["heldout_test"]["model"]["per_class"].items():
        out.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labml.logs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seeds", type=int, default=80)
    b.add_argument("--out", type=Path, default=REPO / "datasets/chaos-logs/logs.jsonl.gz")
    b.add_argument("--procs", type=int, default=os.cpu_count() or 2)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=REPO / "datasets/chaos-logs/logs.jsonl.gz")
    t.add_argument("--model-out", type=Path, default=REPO / "lab/models/log_classifier.json")
    t.add_argument("--results", type=Path, default=REPO / "results/log_classifier")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.seeds, a.out, a.procs)
    else:
        train(a.data, a.model_out, a.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
