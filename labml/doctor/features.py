"""Summary features of a run for the features doctor.

Two groups, so results can say what each model was allowed to see:

* `curve`: only the loss and accuracy curves (what the image doctor sees, as numbers).
* `telemetry`: what trainers usually log besides the curves: gradient norm, learning
  rate schedule, dead/saturated unit share, optimiser, dataset and model size.

Losses are divided by ln(n_classes), the loss of a uniform guess, so 1.0 means "learned
nothing" on every task.
"""

from __future__ import annotations

import math

import numpy as np

CURVE = ["tl_first", "tl_last", "tl_min", "tl_last_over_first", "tl_drop_at_25", "tl_early_spike", "tl_rough",
         "tl_slope_last", "vl_last", "vl_min", "vl_last_over_min", "vl_argmin", "gap_last", "gap_mean",
         "va_last", "va_max", "ta_last", "acc_gap_last", "diverged", "diverged_at", "tl_plateau"]
TELEMETRY = ["gn_first", "gn_max", "gn_last", "gn_max_over_med", "dead_first", "dead_max", "dead_last",
             "lr_peak", "lr_warm", "opt_sgd", "log_n_train", "log_params", "params_per_row", "n_classes"]
ALL = CURVE + TELEMETRY


def _arr(xs) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in xs], dtype=float)


def _nanlast(a: np.ndarray, default: float = np.nan) -> float:
    f = a[np.isfinite(a)]
    return float(f[-1]) if f.size else default


def features(row: dict) -> dict[str, float]:
    s, cfg = row["series"], row["config"]
    chance = math.log(cfg["n_classes"])
    tl, vl = _arr(s["train_loss"]) / chance, _arr(s["val_loss"]) / chance
    ta, va = _arr(s["train_acc"]), _arr(s["val_acc"])
    gn, lr, dead = _arr(s["grad_norm"]), _arr(s["lr"]), _arr(s["dead_frac"])
    n = len(tl)
    bad = np.where(~np.isfinite(tl))[0]
    div = float(bad.size > 0)
    cut = int(bad[0]) if bad.size else n
    tlf, vlf = tl[:cut], vl[:cut]
    cap = 20.0
    f: dict[str, float] = {}
    first = float(tlf[0]) if cut else cap
    last = _nanlast(tlf, cap)
    f["tl_first"] = min(first, cap)
    f["tl_last"] = min(last, cap)
    f["tl_min"] = min(float(np.nanmin(tlf)) if cut else cap, cap)
    f["tl_last_over_first"] = min(last / max(first, 1e-6), cap)
    q = max(1, cut // 4)
    f["tl_drop_at_25"] = float((first - tlf[min(q, cut - 1)]) / max(first - f["tl_min"], 1e-6)) if cut > 1 else 0.0
    early = tlf[: max(2, cut // 3)]
    f["tl_early_spike"] = min(float(np.nanmax(early) / max(first, 1e-6)), cap) if cut else cap
    d = np.diff(np.log(np.clip(tlf, 1e-6, None))) if cut > 2 else np.zeros(1)
    f["tl_rough"] = float(np.std(d))
    tail = tlf[int(cut * 0.8):] if cut > 5 else tlf
    f["tl_slope_last"] = float(np.polyfit(np.arange(tail.size), tail, 1)[0]) if tail.size > 2 else 0.0
    f["tl_plateau"] = float(np.mean(np.abs(d[-10:]) < 0.01)) if d.size >= 10 else 0.0
    vmin = float(np.nanmin(vlf)) if cut and np.isfinite(vlf).any() else cap
    vlast = _nanlast(vlf, cap)
    f["vl_last"], f["vl_min"] = min(vlast, cap), min(vmin, cap)
    f["vl_last_over_min"] = min(vlast / max(vmin, 1e-6), cap)
    f["vl_argmin"] = float(np.nanargmin(vlf)) / n if cut and np.isfinite(vlf).any() else 1.0
    gap = vlf - tlf
    f["gap_last"] = float(np.clip(_nanlast(gap, 0.0), -cap, cap))
    f["gap_mean"] = float(np.clip(np.nanmean(gap), -cap, cap)) if cut and np.isfinite(gap).any() else 0.0
    f["va_last"] = _nanlast(va, 0.0)
    f["va_max"] = float(np.nanmax(va)) if np.isfinite(va).any() else 0.0
    f["ta_last"] = _nanlast(ta, 0.0)
    f["acc_gap_last"] = f["ta_last"] - f["va_last"]
    f["diverged"] = div
    f["diverged_at"] = cut / n
    g = gn[:cut]
    g = g[np.isfinite(g)]
    lg = np.log10(np.clip(g, 1e-8, None)) if g.size else np.array([8.0])
    f["gn_first"], f["gn_max"], f["gn_last"] = float(lg[0]), float(lg.max()), float(lg[-1])
    f["gn_max_over_med"] = float(lg.max() - np.median(lg))
    dd = dead[:cut]
    dd = dd[np.isfinite(dd)]
    f["dead_first"] = float(dd[0]) if dd.size else 1.0
    f["dead_max"] = float(dd.max()) if dd.size else 1.0
    f["dead_last"] = float(dd[-1]) if dd.size else 1.0
    f["lr_peak"] = float(np.log10(np.nanmax(lr)))
    f["lr_warm"] = float(lr[0] / np.nanmax(lr))
    f["opt_sgd"] = float(cfg["opt"] == "sgd")
    f["log_n_train"] = math.log10(cfg["n_train"])
    f["log_params"] = math.log10(row["n_params"])
    f["params_per_row"] = math.log10(row["n_params"] / cfg["n_train"])
    f["n_classes"] = float(cfg["n_classes"])
    return f


def matrix(rows: list[dict], names: list[str] = ALL) -> np.ndarray:
    out = np.array([[features(r)[k] for k in names] for r in rows], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0)
