"""Tiny CPU training runs with known-bad settings: the training-run doctor's data source.

Each run trains a small numpy MLP on a synthetic classification task and logs what a
real trainer logs (train/val loss and accuracy, gradient norm, learning rate, the share
of dead or saturated hidden units) at 50 evenly spaced points. The cause is the setting
that was injected, so the label is exact.

Causes and how they are injected (all other settings come from the healthy recipe):

  healthy            good LR for the optimiser, 5-10% warmup, clean data
  lr_too_high        LR x 30-300
  lr_too_low         LR / 100-1000
  no_warmup          deep net, LR x 4-10, no warmup (the setting warmup exists for)
  label_noise        30-50% of train and val labels flipped (test stays clean)
  overfitting        50-150 training rows, wide net, no weight decay
  train_test_leak    50-90% of the validation rows are copies of training rows
  dead_activations   ReLU with hidden biases initialised to -1.5..-4
  exploding_gradients  6-10 layers, init gain 2.5-4, no clipping
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

CAUSES = ("healthy", "lr_too_high", "lr_too_low", "no_warmup", "label_noise", "overfitting",
          "train_test_leak", "dead_activations", "exploding_gradients")
TASKS = ("blobs", "spiral", "teacher", "xor")
HELDOUT_TASKS = ("rings",)
N_LOG = 50
SERIES = ("train_loss", "val_loss", "train_acc", "val_acc", "grad_norm", "lr", "dead_frac")


@dataclass(frozen=True)
class RunConfig:
    cause: str
    task: str
    task_seed: int
    seed: int
    n_train: int = 2000
    n_val: int = 500
    n_test: int = 1000
    dim: int = 16
    n_classes: int = 4
    depth: int = 2
    width: int = 64
    act: str = "relu"            # relu | tanh | leaky
    init_gain: float = 1.0
    bias_init: float = 0.0
    opt: str = "adam"            # adam | sgd
    lr: float = 2e-3
    warmup_frac: float = 0.05
    steps: int = 400
    batch: int = 64
    weight_decay: float = 1e-4
    clip: float = 0.0            # 0 = off
    label_noise: float = 0.0
    leak_frac: float = 0.0
    # fixes the doctor can switch on
    early_stop: bool = False
    filter_frac: float = 0.0     # small-loss filtering: retrain without the highest-loss rows
    dedup_val: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---- tasks ---------------------------------------------------------------------------------


def make_task(task: str, task_seed: int, n: int, dim: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """n rows of a synthetic task. The same (task, task_seed) always gives the same distribution."""
    g = np.random.default_rng(task_seed)
    rot = np.linalg.qr(g.normal(size=(dim, dim)))[0]
    teacher = [g.normal(size=(dim, 32)) / math.sqrt(dim), g.normal(size=(32, k)) / math.sqrt(32)]
    centers = g.normal(size=(k, dim)) * 1.6
    s = np.random.default_rng(task_seed * 7919 + n)  # rows differ per draw size; callers pass a unique n
    if task == "blobs":
        y = s.integers(0, k, n)
        x = centers[y] + s.normal(size=(n, dim))
    elif task == "teacher":
        x = s.normal(size=(n, dim))
        y = np.argmax(np.tanh(x @ teacher[0]) @ teacher[1], axis=1)
    elif task == "spiral":
        y = s.integers(0, k, n)
        t = s.uniform(0.2, 1, n)
        ang = 2 * math.pi * y / k + 3.5 * t
        p = np.stack([t * np.cos(ang), t * np.sin(ang)], 1) * 3 + s.normal(scale=0.12, size=(n, 2))
        x = np.concatenate([p, s.normal(scale=0.3, size=(n, dim - 2))], 1) @ rot
    elif task == "xor":
        z = s.normal(size=(n, dim))
        bits = (z[:, : max(2, int(math.log2(k)) + 1)] > 0).astype(int)
        y = (bits.sum(1) % 2) if k == 2 else (bits[:, 0] * 2 + bits[:, 1]) % k
        x = z @ rot
    elif task == "rings":
        y = s.integers(0, k, n)
        r = 1 + y + s.normal(scale=0.25, size=n)
        a = s.uniform(0, 2 * math.pi, n)
        p = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        x = np.concatenate([p, s.normal(scale=0.3, size=(n, dim - 2))], 1) @ rot
    else:
        raise ValueError(task)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x.astype(np.float32), y.astype(np.int64)


def make_data(cfg: RunConfig) -> dict[str, np.ndarray]:
    total = cfg.n_train + cfg.n_val + cfg.n_test
    x, y = make_task(cfg.task, cfg.task_seed, total, cfg.dim, cfg.n_classes)
    # the split is fixed by the task, so a fix rerun sees the same rows
    xtr, ytr = x[: cfg.n_train], y[: cfg.n_train].copy()
    xva, yva = x[cfg.n_train: cfg.n_train + cfg.n_val].copy(), y[cfg.n_train: cfg.n_train + cfg.n_val].copy()
    xte, yte = x[cfg.n_train + cfg.n_val:], y[cfg.n_train + cfg.n_val:]
    g = np.random.default_rng(cfg.task_seed + 1)
    if cfg.label_noise > 0:
        for yy in (ytr, yva):
            flip = g.random(len(yy)) < cfg.label_noise
            yy[flip] = (yy[flip] + g.integers(1, cfg.n_classes, flip.sum())) % cfg.n_classes
    if cfg.leak_frac > 0:
        m = int(cfg.n_val * cfg.leak_frac)
        idx = g.choice(cfg.n_train, size=m, replace=m > cfg.n_train)
        xva[:m], yva[:m] = xtr[idx], ytr[idx]
    if cfg.dedup_val:
        seen = {row.tobytes() for row in xtr}
        keep = np.array([row.tobytes() not in seen for row in xva])
        xva, yva = xva[keep], yva[keep]
    return {"xtr": xtr, "ytr": ytr, "xva": xva, "yva": yva, "xte": xte, "yte": yte}


# ---- model ---------------------------------------------------------------------------------


def _act(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return np.maximum(z, 0)
    if kind == "leaky":
        return np.where(z > 0, z, 0.01 * z)
    return np.tanh(z)


def _dact(z: np.ndarray, h: np.ndarray, kind: str) -> np.ndarray:
    if kind == "relu":
        return (z > 0).astype(z.dtype)
    if kind == "leaky":
        return np.where(z > 0, 1.0, 0.01).astype(z.dtype)
    return 1 - h * h


class MLP:
    def __init__(self, cfg: RunConfig, rng: np.random.Generator):
        self.cfg = cfg
        sizes = [cfg.dim] + [cfg.width] * cfg.depth + [cfg.n_classes]
        self.params: list[np.ndarray] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            base = 2.0 if cfg.act in ("relu", "leaky") else 1.0
            gain = cfg.init_gain if i < len(sizes) - 2 else 1.0
            w = rng.normal(scale=gain * math.sqrt(base / a), size=(a, b)).astype(np.float32)
            bias = np.full(b, cfg.bias_init if i < len(sizes) - 2 else 0.0, dtype=np.float32)
            self.params += [w, bias]

    def forward(self, x: np.ndarray):
        cache = []
        h = x
        n_layers = len(self.params) // 2
        with np.errstate(all="ignore"):
            for i in range(n_layers):
                w, b = self.params[2 * i], self.params[2 * i + 1]
                z = h @ w + b
                if i < n_layers - 1:
                    hn = _act(z, self.cfg.act)
                    cache.append((h, z, hn))
                    h = hn
                else:
                    cache.append((h, z, None))
                    return z, cache
        raise AssertionError

    def loss_grad(self, x: np.ndarray, y: np.ndarray):
        logits, cache = self.forward(x)
        with np.errstate(all="ignore"):
            m = logits.max(1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
            loss = float(np.mean(lse - logits[np.arange(len(y)), y]))
            p = np.exp(logits - lse[:, None])
            d = p
            d[np.arange(len(y)), y] -= 1
            d /= len(y)
            grads: list[np.ndarray] = [None] * len(self.params)  # type: ignore[list-item]
            for i in range(len(cache) - 1, -1, -1):
                h_in, z, h_out = cache[i]
                if h_out is not None:
                    d = d * _dact(z, h_out, self.cfg.act)
                grads[2 * i] = h_in.T @ d
                grads[2 * i + 1] = d.sum(0)
                if i:
                    d = d @ self.params[2 * i].T
        return loss, grads

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """Loss, accuracy and the share of dead (ReLU) or saturated (tanh) hidden units."""
        if len(y) == 0:
            return float("nan"), float("nan"), float("nan")
        logits, cache = self.forward(x)
        with np.errstate(all="ignore"):
            m = logits.max(1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
            loss = float(np.mean(lse - logits[np.arange(len(y)), y]))
            acc = float(np.mean(np.argmax(logits, 1) == y)) if np.all(np.isfinite(logits)) else float("nan")
            dead = []
            for _, z, h in cache[:-1]:
                dead.append(np.all(z <= 0, axis=0) if self.cfg.act != "tanh" else np.mean(np.abs(h) > 0.99, 0) > 0.9)
        return loss, acc, float(np.mean(np.concatenate(dead))) if dead else 0.0

    def per_example_loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        with np.errstate(all="ignore"):
            m = logits.max(1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(logits - m).sum(1))
            return lse - logits[np.arange(len(y)), y]


def lr_at(cfg: RunConfig, step: int) -> float:
    w = int(cfg.steps * cfg.warmup_frac)
    if w and step < w:
        return cfg.lr * (step + 1) / w
    t = (step - w) / max(1, cfg.steps - w)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def _train(cfg: RunConfig, xtr, ytr, xva, yva, rng) -> tuple[MLP, dict[str, list[float]], MLP | None]:
    model = MLP(cfg, rng)
    state = [np.zeros_like(p) for p in model.params]
    state2 = [np.zeros_like(p) for p in model.params]
    hist: dict[str, list[float]] = {k: [] for k in SERIES}
    every = cfg.steps // N_LOG
    sub = rng.choice(len(ytr), size=min(512, len(ytr)), replace=False)
    run_loss, run_gn, n_acc = 0.0, 0.0, 0
    best, best_val = None, float("inf")
    diverged = False
    for step in range(cfg.steps):
        idx = rng.integers(0, len(ytr), cfg.batch)
        if not diverged:
            loss, grads = model.loss_grad(xtr[idx], ytr[idx])
            with np.errstate(all="ignore"):
                gn = float(math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads)))
            if not math.isfinite(loss) or not math.isfinite(gn) or loss > 1e4:
                diverged = True
            else:
                if cfg.clip and gn > cfg.clip:
                    grads = [g * (cfg.clip / gn) for g in grads]
                lr = lr_at(cfg, step)
                for i, (p, g) in enumerate(zip(model.params, grads)):
                    if cfg.opt == "adam":
                        state[i] = 0.9 * state[i] + 0.1 * g
                        state2[i] = 0.999 * state2[i] + 0.001 * g * g
                        mhat = state[i] / (1 - 0.9 ** (step + 1))
                        vhat = state2[i] / (1 - 0.999 ** (step + 1))
                        p -= lr * (mhat / (np.sqrt(vhat) + 1e-8) + cfg.weight_decay * p)
                    else:
                        state[i] = 0.9 * state[i] + g
                        p -= lr * (state[i] + cfg.weight_decay * p)
                run_loss += loss
                run_gn += gn
                n_acc += 1
        if (step + 1) % every == 0:
            if diverged:
                for k in SERIES:
                    hist[k].append(float("nan") if k not in ("lr",) else lr_at(cfg, step))
                hist["train_loss"][-1] = hist["val_loss"][-1] = float("nan")
                continue
            vl, va, dead = model.evaluate(xva, yva)
            _, ta, _ = model.evaluate(xtr[sub], ytr[sub])
            hist["train_loss"].append(run_loss / max(1, n_acc))
            hist["grad_norm"].append(run_gn / max(1, n_acc))
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
            hist["train_acc"].append(ta)
            hist["lr"].append(lr_at(cfg, step))
            hist["dead_frac"].append(dead)
            run_loss, run_gn, n_acc = 0.0, 0.0, 0
            if cfg.early_stop and math.isfinite(vl) and vl < best_val:
                best_val = vl
                best = MLP.__new__(MLP)
                best.cfg, best.params = cfg, [p.copy() for p in model.params]
    return model, hist, best


def run(cfg: RunConfig) -> dict:
    """Train once (twice with small-loss filtering) and return metrics + the logged series."""
    d = make_data(cfg)
    rng = np.random.default_rng(cfg.seed)
    xtr, ytr = d["xtr"], d["ytr"]
    if cfg.filter_frac > 0:
        first, _, _ = _train(replace(cfg, filter_frac=0), xtr, ytr, d["xva"], d["yva"], np.random.default_rng(cfg.seed))
        losses = first.per_example_loss(xtr, ytr)
        losses = np.where(np.isfinite(losses), losses, np.inf)
        keep = np.argsort(losses)[: int(len(ytr) * (1 - cfg.filter_frac))]
        xtr, ytr = xtr[keep], ytr[keep]
    model, hist, best = _train(cfg, xtr, ytr, d["xva"], d["yva"], rng)
    final = best if (cfg.early_stop and best is not None) else model
    te_loss, te_acc, _ = final.evaluate(d["xte"], d["yte"])
    va_loss, va_acc, _ = final.evaluate(d["xva"], d["yva"])
    n_params = sum(p.size for p in model.params)
    return {"config": cfg.to_dict(), "series": hist, "n_params": int(n_params), "n_val_used": int(len(d["yva"])),
            "final": {"val_loss": va_loss, "val_acc": va_acc, "test_loss": te_loss, "test_acc": te_acc,
                      "diverged": bool(any(not math.isfinite(v) for v in hist["train_loss"]))}}


# ---- sampling configs ----------------------------------------------------------------------


def _loguniform(g: np.random.Generator, lo: float, hi: float) -> float:
    return float(math.exp(g.uniform(math.log(lo), math.log(hi))))


def healthy_lr(opt: str, g: np.random.Generator) -> float:
    return _loguniform(g, 1e-3, 4e-3) if opt == "adam" else _loguniform(g, 0.02, 0.08)


def sample_config(cause: str, task: str, seed: int) -> RunConfig:
    """A run config for `cause`. Same arguments, same config."""
    g = np.random.default_rng([CAUSES.index(cause), TASKS.index(task) if task in TASKS else 99, seed])
    opt = "adam" if g.random() < 0.7 else "sgd"
    k = int(g.choice([2, 3, 4, 5, 6])) if task != "xor" else int(g.choice([2, 4]))
    cfg = RunConfig(cause=cause, task=task, task_seed=int(g.integers(0, 10**6)), seed=int(g.integers(0, 10**6)),
                    n_train=int(g.integers(1000, 4001)), dim=int(g.choice([8, 16, 32])), n_classes=k,
                    depth=int(g.integers(1, 5)), width=int(g.choice([32, 64, 128])),
                    act="relu" if g.random() < 0.7 else "tanh", opt=opt, lr=healthy_lr(opt, g),
                    warmup_frac=float(g.uniform(0.05, 0.1)), weight_decay=1e-4)
    if cause == "lr_too_high":
        cfg = replace(cfg, lr=cfg.lr * _loguniform(g, 30, 300))
    elif cause == "lr_too_low":
        cfg = replace(cfg, lr=cfg.lr / _loguniform(g, 100, 1000))
    elif cause == "no_warmup":
        cfg = replace(cfg, depth=int(g.integers(4, 7)), width=int(g.choice([64, 128])), warmup_frac=0.0,
                      lr=cfg.lr * _loguniform(g, 4, 10))
    elif cause == "label_noise":
        cfg = replace(cfg, label_noise=float(g.uniform(0.3, 0.5)))
    elif cause == "overfitting":
        cfg = replace(cfg, n_train=int(g.integers(50, 151)), width=256, depth=int(g.integers(2, 4)), weight_decay=0.0)
    elif cause == "train_test_leak":
        cfg = replace(cfg, n_train=int(g.integers(100, 301)), width=int(g.choice([128, 256])),
                      leak_frac=float(g.uniform(0.5, 0.9)), weight_decay=0.0)
    elif cause == "dead_activations":
        cfg = replace(cfg, act="relu", bias_init=-float(g.uniform(1.5, 4.0)))
    elif cause == "exploding_gradients":
        cfg = replace(cfg, depth=int(g.integers(6, 11)), act="relu", init_gain=float(g.uniform(2.5, 4.0)),
                      opt="sgd", lr=healthy_lr("sgd", g), clip=0.0)
    return cfg


# ---- fixes ---------------------------------------------------------------------------------


def apply_fix(cfg: RunConfig, diagnosis: str) -> RunConfig:
    """The setting change the doctor makes for a diagnosis. Repeated calls keep adjusting."""
    if diagnosis == "lr_too_high":
        return replace(cfg, lr=cfg.lr / 10)
    if diagnosis == "lr_too_low":
        return replace(cfg, lr=cfg.lr * 10)
    if diagnosis == "no_warmup":
        return replace(cfg, warmup_frac=0.1)
    if diagnosis == "label_noise":
        return replace(cfg, filter_frac=min(0.5, cfg.filter_frac + 0.35))
    if diagnosis == "overfitting":
        return replace(cfg, weight_decay=max(cfg.weight_decay, 0.05), early_stop=True)
    if diagnosis == "train_test_leak":
        return replace(cfg, dedup_val=True)
    if diagnosis == "dead_activations":
        return replace(cfg, bias_init=0.0, act="leaky" if cfg.act == "relu" else cfg.act)
    if diagnosis == "exploding_gradients":
        return replace(cfg, init_gain=1.0, clip=1.0)
    return cfg


def reference_config(cfg: RunConfig) -> RunConfig:
    """The healthy recipe on the same task, rows and size: the bar a fixed run must reach."""
    g = np.random.default_rng(cfg.seed)
    return replace(cfg, cause="reference", lr=healthy_lr(cfg.opt, g) if cfg.opt == "adam" else 0.05,
                   warmup_frac=0.08, label_noise=0.0, leak_frac=0.0, bias_init=0.0, init_gain=1.0,
                   depth=min(cfg.depth, 4), width=min(cfg.width, 128), weight_decay=1e-4, clip=1.0,
                   early_stop=True, filter_frac=0.0, dedup_val=False)
