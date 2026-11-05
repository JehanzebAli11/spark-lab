"""python -m guard.media --out runs/media [--dataset DIR] [--firewall DIR ...] [--only post1,post2,...]

Builds every LinkedIn asset from run directories:
  post1-dataset/    carousel.pdf + slide PNGs + grid.png       (from a screen_injection run)
  post2-wallet/     carousel.pdf + slide PNGs + terminal.mp4   (from the wallet decoder itself)
  post3-extension/  extension-demo.mp4 + before/after/phishing (the real extension in Chromium)
  post4-firewall/   chart-by-technique.png, chart-by-goal.png, firewall-replay.mp4 (from firewall runs)

Without --dataset / --firewall it makes small local runs in OUT/_work first. On the Spark, point it at
the real job outputs (runs/screen-injection/v1, runs/firewall/...) so the numbers on every graphic are
the real ones. The first --firewall dir is the baseline (usually --guard none).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

POSTS = ("post1", "post2", "post3", "post4")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m guard.media")
    ap.add_argument("--out", type=Path, default=Path("runs/media"))
    ap.add_argument("--dataset", type=Path, help="a guard.screen_injection output directory")
    ap.add_argument("--firewall", type=Path, nargs="*", help="guard.firewall output dirs; the first is the baseline")
    ap.add_argument("--only", default=",".join(POSTS), help="comma-separated subset of " + ",".join(POSTS))
    a = ap.parse_args(argv)
    only = set(a.only.split(","))
    work = a.out / "_work"

    from playwright.sync_api import sync_playwright

    from . import dataset, extension, firewall, wallet

    if "post1" in only and not a.dataset:
        from ..screen_injection.__main__ import main as gen
        a.dataset = work / "dataset"
        gen(["--n", "600", "--out", str(a.dataset), "--preview", "0"])
    if "post4" in only and not a.firewall:
        from ..firewall.__main__ import main as fw
        a.firewall = []
        for g in ("none", "rules"):
            d = work / f"firewall-{g}"
            fw(["--out", str(d), "--guard", g, "--per-task", "120"])
            a.firewall.append(d)

    made = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        if "post1" in only:
            made += dataset.slides(a.dataset, b, a.out / "post1-dataset")
            made.append(dataset.grid(a.dataset, b, a.out / "post1-dataset"))
        if "post2" in only:
            made += wallet.slides(b, a.out / "post2-wallet")
            made.append(wallet.video(b, a.out / "post2-wallet"))
        if "post4" in only:
            o = a.out / "post4-firewall"
            made += [firewall.chart(a.firewall, b, o, "technique"), firewall.chart(a.firewall, b, o, "goal")]
            if len(a.firewall) >= 2:
                r = firewall.replay(a.firewall[0], a.firewall[-1], b, o)
                if r:
                    made.append(r)
        b.close()
        if "post3" in only:
            made += list(extension.video(p, a.out / "post3-extension").values())
    for m in made:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
