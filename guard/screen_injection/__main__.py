"""python -m guard.screen_injection --n 5000 --out runs/screen-injection/v1 [--seed 0] [--preview 20]

Renders pages into OUT/images, OUT/pages and OUT/previews, and writes OUT/annotations.jsonl,
OUT/summary.json and OUT/summary.md (the nightly report quotes summary.md).
Safe to rerun: pages already in annotations.jsonl are skipped, so a killed job resumes.
`--html-only` skips the browser and writes OUT/specs.jsonl plus the HTML (for text agents and tests).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .generate import build_page


def _done_ids(out: Path) -> set[str]:
    f = out / "annotations.jsonl"
    if not f.exists():
        return set()
    ids = set()
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            continue  # a line cut short by a crash; that page is simply rendered again
    return ids


def summarise(out: Path) -> dict:
    recs = [json.loads(x) for x in (out / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    objs = [o for r in recs for o in r["objects"]]
    inj = [o for o in objs if o["label"] == "injection"]
    s = {
        "pages": len(recs),
        "pages_with_visible_injection": sum(r["has_visible_injection"] for r in recs),
        "pages_with_dom_injection": sum(r["has_dom_injection"] for r in recs),
        "clean_pages": sum(not any(o["label"] == "injection" for o in r["objects"]) for r in recs),
        "injections": len(inj),
        "injections_on_screen": sum(o["on_screen"] for o in inj),
        "decoys": sum(o["label"] == "decoy" for o in objs),
        "by_technique": dict(Counter(o["technique"] for o in inj)),
        "by_goal": dict(Counter(o["goal"] for o in inj)),
        "by_template": dict(Counter(r["template"] for r in recs)),
        "roman_urdu_injections": sum(o["lang"] == "roman_urdu" for o in inj),
    }
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(s["by_technique"].items()))
    (out / "summary.md").write_text(
        f"Screen-injection dataset: {s['pages']} pages, {s['injections']} injections "
        f"({s['injections_on_screen']} on screen), {s['decoys']} decoys, {s['clean_pages']} clean pages.\n\n"
        f"| Technique | Injections |\n| --- | --- |\n{rows}\n")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m guard.screen_injection")
    ap.add_argument("--n", type=int, default=1000, help="number of pages")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="first seed; page i uses seed + i")
    ap.add_argument("--preview", type=int, default=20, help="also save this many pages with boxes drawn")
    ap.add_argument("--clean-rate", type=float, default=0.25, help="share of pages with no injection")
    ap.add_argument("--html-only", action="store_true", help="skip rendering; write specs and HTML only")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    specs = (build_page(a.seed + i, clean_rate=a.clean_rate) for i in range(a.n))
    if a.html_only:
        (a.out / "pages").mkdir(exist_ok=True)
        with open(a.out / "specs.jsonl", "w", encoding="utf-8") as f:
            for s in specs:
                (a.out / "pages" / f"{s.id}.html").write_text(s.html, encoding="utf-8")
                f.write(json.dumps(s.to_dict()) + "\n")
        print(f"wrote {a.n} page specs to {a.out}")
        return 0

    from .render import render

    done = _done_ids(a.out)
    todo = [s for s in specs if s.id not in done]
    print(f"{len(done)} pages already rendered, {len(todo)} to go", flush=True)
    render(todo, a.out, preview=a.preview if not done else 0)
    s = summarise(a.out)
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
