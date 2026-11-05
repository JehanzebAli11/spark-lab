"""Render a run's curves as a small picture: what the image doctor (and later a VLM) sees.

Blue: train loss, orange: val loss (shared log axis). Green dashed: val accuracy on a 0-1
right axis. A red cross marks where the loss became NaN/inf. No text, no ticks.
"""

from __future__ import annotations

import io
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SIZE = 64


def _figure(series: dict, w_px: int = 160, h_px: int = 120):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w_px / 40, h_px / 40), dpi=40)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    tl = np.array([np.nan if v is None else v for v in series["train_loss"]], dtype=float)
    vl = np.array([np.nan if v is None else v for v in series["val_loss"]], dtype=float)
    va = np.array([np.nan if v is None else v for v in series["val_acc"]], dtype=float)
    x = np.arange(len(tl))
    finite = np.concatenate([tl[np.isfinite(tl)], vl[np.isfinite(vl)]])
    finite = finite[finite > 0]
    lo, hi = (finite.min(), finite.max()) if finite.size else (1e-3, 10)
    lo, hi = max(lo * 0.8, 1e-4), max(hi * 1.25, lo * 10)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(-1, len(tl))
    ax.plot(x, np.clip(tl, lo, hi), color="#1f5fbf", lw=2.2)
    ax.plot(x, np.clip(vl, lo, hi), color="#f08a00", lw=2.2)
    bad = np.where(~np.isfinite(tl))[0]
    if bad.size:
        ax.plot([bad[0]], [math.sqrt(lo * hi)], marker="x", color="#d00000", ms=14, mew=4)
    ax2 = ax.twinx()
    ax2.set_ylim(-0.02, 1.02)
    ax2.plot(x, va, color="#1a9a3a", lw=1.6, ls="--")
    for a in (ax, ax2):
        a.set_xticks([])
        a.set_yticks([])
        a.minorticks_off()
    return fig, plt


def png_bytes(series: dict, w_px: int = 160, h_px: int = 120) -> bytes:
    fig, plt = _figure(series, w_px, h_px)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=40)
    plt.close(fig)
    return buf.getvalue()


def to_array(png: bytes, size: int = SIZE) -> np.ndarray:
    from PIL import Image

    im = Image.open(io.BytesIO(png)).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _one(args: tuple[dict, str | None]) -> np.ndarray:
    row, path = args
    png = png_bytes(row["series"])
    if path:
        Path(path).write_bytes(png)
    return to_array(png)


def render_all(rows: list[dict], out_dir: Path | None, procs: int = 2) -> np.ndarray:
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, str(out_dir / f"{r['id']}.png") if out_dir else None) for r in rows]
    with Pool(procs) as pool:
        arrs = pool.map(_one, jobs, chunksize=16)
    return np.stack(arrs)
