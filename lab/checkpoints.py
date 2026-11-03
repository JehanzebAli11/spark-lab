"""Find and prune training checkpoints.

Convention (matches Hugging Face Trainer / TRL): a run directory contains
`checkpoint-<step>/` folders. The newest by step number is the resume point.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

_CKPT = re.compile(r"^checkpoint-(\d+)$")


def checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoint dirs in run_dir, oldest first by step."""
    if not run_dir.is_dir():
        return []
    found = []
    for p in run_dir.iterdir():
        m = _CKPT.match(p.name)
        if m and p.is_dir():
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def latest_checkpoint(run_dir: Path) -> Path | None:
    cps = checkpoints(run_dir)
    return cps[-1] if cps else None


def prune(root: Path, keep: int = 3, dry_run: bool = False) -> list[Path]:
    """Keep the newest `keep` checkpoints in every run dir under root. Returns removed paths.

    Refuses to touch anything outside root or any path containing eval/locked.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    root = root.resolve()
    removed: list[Path] = []
    run_dirs = {p.parent for p in root.rglob("checkpoint-*") if p.is_dir()}
    for run_dir in sorted(run_dirs):
        for cp in checkpoints(run_dir)[:-keep]:
            cp = cp.resolve()
            if root not in cp.parents or "eval/locked" in cp.as_posix():
                continue
            removed.append(cp)
            if not dry_run:
                shutil.rmtree(cp)
    return removed
