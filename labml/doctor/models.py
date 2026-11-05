"""Train and evaluate the training-run doctors.

    python -m labml.doctor.models

Doctors (all CPU):
  image      small CNN on the 64x64 curve picture only
  curves     gradient boosting on curve-shape features only (same information as the image)
  features   gradient boosting on curve + telemetry features (grad norm, LR, dead units, sizes)
  ensemble   mean of image and features probabilities

Rows: healthy runs plus injected runs that were actually harmed (see dataset.py). Injected
runs that came out fine carry no symptom to diagnose; they are counted, not trained on.

Evaluation: test (unseen seeds), heldout_task (unseen task family), and leave-one-cause-out
open-set runs: train without cause C and check whether C's runs get escalated (low
confidence) rather than confidently mislabelled. Threshold per doctor: lowest confidence
with >= 90% precision-when-acting on val.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .features import ALL, CURVE, matrix  # noqa: E402
from .sim import CAUSES  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "datasets" / "doctor-runs"
RESULTS = REPO / "results" / "doctor"
MODELS = REPO / "data" / "doctor" / "models"
SPLITS = ("train", "val", "test", "heldout_task")


def load(data: Path = DATA) -> tuple[list[dict], np.ndarray]:
    with gzip.open(data / "runs.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    z = np.load(data / "images.npz")
    assert list(z["ids"]) == [r["id"] for r in rows]
    return rows, z["images"]


def usable(rows: list[dict]) -> np.ndarray:
    return np.array([r["cause"] == "healthy" or r["harmed"] for r in rows])


# ---- models ------------------------------------------------------------------------------


def fit_gbm(x: np.ndarray, y: np.ndarray, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.12, random_state=seed)
    return m.fit(x, y)


class CNN:
    """3x64x64 -> 3 conv blocks -> 2 dense layers."""

    def __init__(self, n_classes: int, seed: int = 0):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch = torch
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, n_classes))
        self.classes_: np.ndarray | None = None

    def _x(self, imgs: np.ndarray):
        return self.torch.from_numpy(imgs.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)

    def fit(self, imgs: np.ndarray, y: np.ndarray, epochs: int = 25, xval=None, yval=None, log=print) -> CNN:
        torch = self.torch
        torch.set_num_threads(os.cpu_count() or 2)
        self.classes_ = np.array(sorted(set(y)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        x, t = self._x(imgs), torch.tensor([idx[c] for c in y])
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(t) + 63) // 64))
        g = torch.Generator().manual_seed(0)
        best, best_acc = None, -1.0
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(t), generator=g)
            tot = 0.0
            for i in range(0, len(t), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.net(x[b]), t[b])
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * len(b)
            if xval is not None:
                acc = float(np.mean(self.classes_[self.predict_proba(xval).argmax(1)] == yval))
                if acc > best_acc:
                    best_acc, best = acc, {k: v.clone() for k, v in self.net.state_dict().items()}
                log(f"  cnn epoch {ep + 1}/{epochs} loss {tot / len(t):.3f} val acc {acc:.3f}")
        if best is not None:
            self.net.load_state_dict(best)
        return self

    def predict_proba(self, imgs: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 256):
                out.append(torch.softmax(self.net(self._x(imgs[i:i + 256])), 1).numpy())
        return np.concatenate(out)


# ---- metrics -----------------------------------------------------------------------------


def metrics(gold, pred, classes) -> dict:
    gold, pred = np.asarray(gold), np.asarray(pred)
    acting = pred != "unknown"
    per = {}
    for c in classes:
        tp = int(np.sum((gold == c) & (pred == c)))
        fp = int(np.sum((gold != c) & (pred == c)))
        fn = int(np.sum((gold == c) & (pred != c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0,
                  "support": tp + fn}
    return {"n": int(len(gold)), "accuracy": round(float(np.mean(gold == pred)), 4),
            "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4),
            "coverage": round(float(acting.mean()), 4),
            "precision_when_acting": round(float(np.mean(gold[acting] == pred[acting])), 4) if acting.any() else 0.0,
            "per_class": per}


def threshold_for(probs: np.ndarray, gold: np.ndarray, classes: np.ndarray, target: float = 0.9) -> float:
    conf, lab = probs.max(1), classes[probs.argmax(1)]
    for t in np.arange(0.2, 0.99, 0.01):
        keep = conf >= t
        if keep.any() and np.mean(lab[keep] == gold[keep]) >= target:
            return round(float(t), 2)
    return 0.99


def label(probs: np.ndarray, classes: np.ndarray, t: float = 0.0) -> np.ndarray:
    lab = classes[probs.argmax(1)].astype(object)
    lab[probs.max(1) < t] = "unknown"
    return lab


# ---- main --------------------------------------------------------------------------------


def train_all(rows, imgs, results: Path = RESULTS, epochs: int = 25) -> dict:
    results.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    ok = usable(rows)
    split = np.array([r["split"] for r in rows])
    y = np.array([r["cause"] for r in rows])
    xf, xc = matrix(rows, ALL), matrix(rows, CURVE)
    S = {s: np.where(ok & (split == s))[0] for s in SPLITS}
    tr, va = S["train"], S["val"]
    report: dict = {"counts": {s: dict(Counter(y[i])) for s, i in S.items()},
                    "injected_but_unharmed": dict(Counter(y[~ok & (y != "healthy")])),
                    "doctors": {}}

    print("fitting features + curves GBMs", flush=True)
    gbm_f = fit_gbm(xf[tr], y[tr])
    gbm_c = fit_gbm(xc[tr], y[tr])
    print("fitting CNN", flush=True)
    cnn = CNN(len(CAUSES)).fit(imgs[tr], y[tr], epochs=epochs, xval=imgs[va], yval=y[va])

    def probs_of(name, idx):
        if name == "features":
            return gbm_f.predict_proba(xf[idx])
        if name == "curves":
            return gbm_c.predict_proba(xc[idx])
        if name == "image":
            return cnn.predict_proba(imgs[idx])
        return (gbm_f.predict_proba(xf[idx]) + cnn.predict_proba(imgs[idx])) / 2

    classes = np.array(gbm_f.classes_)
    assert list(classes) == list(cnn.classes_) == list(gbm_c.classes_)
    for name in ("image", "curves", "features", "ensemble"):
        t = threshold_for(probs_of(name, va), y[va], classes)
        rep = {"threshold": t}
        for s in ("val", "test", "heldout_task"):
            p = probs_of(name, S[s])
            rep[s] = {"raw": metrics(y[S[s]], label(p, classes), CAUSES),
                      "thresholded": metrics(y[S[s]], label(p, classes, t), CAUSES)}
        report["doctors"][name] = rep
        print(f"{name:9s} test acc {rep['test']['raw']['accuracy']:.3f}  heldout_task "
              f"{rep['heldout_task']['raw']['accuracy']:.3f}  threshold {t}", flush=True)

    # leave-one-cause-out open set (features GBM; cheap)
    print("open-set: leave one cause out", flush=True)
    t_f = report["doctors"]["features"]["threshold"]
    open_set = {}
    for c in CAUSES:
        if c == "healthy":
            continue
        keep = tr[y[tr] != c]
        m = fit_gbm(xf[keep], y[keep])
        test_c = np.where(ok & (y == c) & np.isin(split, ["test", "heldout_task"]))[0]
        p = m.predict_proba(xf[test_c])
        lab = label(p, np.array(m.classes_), t_f)
        cnt = Counter(lab)
        open_set[c] = {"n": int(len(test_c)), "escalated": round(cnt.get("unknown", 0) / max(1, len(test_c)), 3),
                       "called_healthy": round(cnt.get("healthy", 0) / max(1, len(test_c)), 3),
                       "most_common_wrong": [(k, v) for k, v in cnt.most_common() if k != "unknown"][:2]}
    report["open_set"] = open_set

    # confusion for the ensemble and the feature importances proxy
    t_e = report["doctors"]["ensemble"]["threshold"]
    _confusion(y[S["test"]], label(probs_of("ensemble", S["test"]), classes, t_e),
               results / "confusion_ensemble_test.png", "ensemble, test (unknown = escalated)")
    _confusion(y[S["heldout_task"]], label(probs_of("features", S["heldout_task"]), classes),
               results / "confusion_features_heldout_task.png", "features doctor, held-out task family")
    with open(MODELS / "doctors.pkl", "wb") as fh:
        pickle.dump({"features": gbm_f, "curves": gbm_c, "classes": classes,
                     "thresholds": {k: v["threshold"] for k, v in report["doctors"].items()}}, fh)
    cnn.torch.save(cnn.net.state_dict(), MODELS / "cnn.pt")
    (results / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (results / "report.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    return {"report": report, "gbm_f": gbm_f, "cnn": cnn, "classes": classes}


def _confusion(gold, pred, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = list(CAUSES) + (["unknown"] if "unknown" in set(pred) else [])
    m = np.array([[np.sum((gold == g) & (pred == c)) for c in cols] for g in CAUSES])
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.imshow(m, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j]:
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(CAUSES)), CAUSES)
    ax.set_xlabel("diagnosis")
    ax.set_ylabel("injected cause")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def markdown(rep: dict) -> str:
    out = ["# Training-run doctor results", "",
           "Rows: healthy runs + injected runs that were actually harmed. Injected runs that came out fine "
           f"(not used): {rep['injected_but_unharmed']}.", "",
           "| Doctor | Sees | Test acc | Test macro-F1 | Held-out task acc | Held-out task macro-F1 | Threshold "
           "| Test: acts on | Test: precision when acting |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    sees = {"image": "curve picture", "curves": "curve numbers", "features": "curves + telemetry",
            "ensemble": "image + features"}
    for name, d in rep["doctors"].items():
        t, h, th = d["test"]["raw"], d["heldout_task"]["raw"], d["test"]["thresholded"]
        out.append(f"| {name} | {sees[name]} | {t['accuracy']:.3f} | {t['macro_f1']:.3f} | {h['accuracy']:.3f} | "
                   f"{h['macro_f1']:.3f} | {d['threshold']:.2f} | {th['coverage']:.1%} | "
                   f"{th['precision_when_acting']:.3f} |")
    out += ["", "Per-cause F1 on test:", "", "| Cause | " + " | ".join(rep["doctors"]) + " |",
            "| --- |" + " --- |" * len(rep["doctors"])]
    for c in CAUSES:
        out.append(f"| {c} | " + " | ".join(f"{d['test']['raw']['per_class'][c]['f1']:.3f}"
                                             for d in rep["doctors"].values()) + " |")
    out += ["", "## Unseen causes (leave-one-cause-out, features doctor)", "",
            "Trained without the cause, tested on its runs: does the doctor escalate or guess?", "",
            "| Held-out cause | n | Escalated | Called healthy | Most common wrong labels |",
            "| --- | --- | --- | --- | --- |"]
    for c, v in rep["open_set"].items():
        wrong = ", ".join(f"{k} ({n})" for k, n in v["most_common_wrong"]) or "-"
        out.append(f"| {c} | {v['n']} | {v['escalated']:.1%} | {v['called_healthy']:.1%} | {wrong} |")
    out += ["", "Rows per split: " + "; ".join(f"{s}: {sum(c.values())}" for s, c in rep["counts"].items()), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labml.doctor.models")
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args(argv)
    rows, imgs = load()
    train_all(rows, imgs, epochs=a.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
