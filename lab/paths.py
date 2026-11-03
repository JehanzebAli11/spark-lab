"""Filesystem layout for the lab. Everything is relative to LAB_ROOT (default: repo root)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabPaths:
    root: Path

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def queue(self) -> Path:
        return self.jobs / "queue"

    @property
    def running(self) -> Path:
        return self.jobs / "running"

    @property
    def done(self) -> Path:
        return self.jobs / "done"

    @property
    def failed(self) -> Path:
        return self.jobs / "failed"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def locked(self) -> Path:
        return self.root / "eval" / "locked"

    @property
    def lockfile(self) -> Path:
        return self.jobs / ".worker.lock"

    def ensure(self) -> LabPaths:
        for d in (self.queue, self.running, self.done, self.failed, self.logs, self.runs, self.reports):
            d.mkdir(parents=True, exist_ok=True)
        return self


def lab_paths(root: str | os.PathLike[str] | None = None) -> LabPaths:
    if root is None:
        root = os.environ.get("LAB_ROOT") or Path(__file__).resolve().parent.parent
    return LabPaths(Path(root).resolve())
