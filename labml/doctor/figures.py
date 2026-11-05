"""Figures for the README: a grid of curve pictures per cause, and rescue rates per policy.

    python -m labml.doctor.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import RESULTS, load, usable
from .sim import CAUSES


def grid(out: Path, per: int = 6) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, imgs = load()
    ok = usable(rows)
    fig, axes = plt.subplots(len(CAUSES), per, figsize=(per * 1.3, len(CAUSES) * 1.3))
    for i, c in enumerate(CAUSES):
        idx = [j for j, r in enumerate(rows) if r["cause"] == c and ok[j] and r["split"] == "test"][:per]
        for k in range(per):
            ax = axes[i, k]
            ax.axis("off")
            if k < len(idx):
                ax.imshow(imgs[idx[k]])
        axes[i, 0].text(-8, 32, c, ha="right", va="center", fontsize=8)
    fig.suptitle("What the image doctor sees (test runs). Blue train loss, orange val loss, green val acc",
                 fontsize=8)
    fig.subplots_adjust(left=0.24, wspace=0.05, hspace=0.05, top=0.95)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def rescue_bars(out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((RESULTS / "autofix.json").read_text())
    causes = list(s["by_cause"])
    pols = ["none", "doctor", "oracle"]
    colors = {"none": "#9aa0a6", "doctor": "#2a6fdb", "oracle": "#1a9a3a"}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.27
    for k, p in enumerate(pols):
        vals = []
        for c in causes:
            a, b = s["by_cause"][c][p].split("/")
            vals.append(int(a) / max(1, int(b)))
        ax.bar(np.arange(len(causes)) + (k - 1) * w, vals, w, label=p, color=colors[p])
    ax.set_xticks(range(len(causes)), causes, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("share of harmed runs rescued")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Auto-fix loop: rerun unchanged vs doctor vs true-cause fix (test + held-out task)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    grid(RESULTS / "curve_grid.png")
    rescue_bars(RESULTS / "rescue_by_cause.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
