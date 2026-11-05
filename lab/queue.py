"""A small file-based job queue that survives reboots and power cuts.

A job is a JSON file. Lifecycle: jobs/queue/ -> jobs/running/ -> jobs/done/ or jobs/failed/.
Moves are atomic renames on one filesystem, so a crash never loses a job: on start the
worker puts anything left in running/ back at the front of the queue, and training jobs
resume from their newest checkpoint.

Job fields:
  name         required, [A-Za-z0-9._-]+
  cmd          required, list of strings (no shell)
  cwd          working directory, relative to LAB_ROOT (default ".")
  priority     lower runs first (default 50)
  timeout_min  kill after this many minutes (default 720)
  retries      extra attempts after a failure (default 1)
  resume_dir   run dir with checkpoint-N/ folders; the newest is passed as LAB_RESUME_FROM
  env          extra environment variables
  report       markdown file (relative to cwd) the job writes; quoted in the nightly report
  notify       send phone alerts for this job (default true)
  heal         classify failures and apply known fixes (default true; see lab/heal.py)
  max_heals    healing retries allowed on top of `retries` (default 4)
  batch_env    env var in `env` holding the batch size; halved on out-of-memory
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .checkpoints import latest_checkpoint
from .heal import Healer
from .notify import notify as default_notify
from .paths import LabPaths
from .triage import UNKNOWN, Diagnosis

_NAME = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class JobError(ValueError):
    pass


def now_iso() -> str:
    """Local time with UTC offset, so reports and logs read in the owner's time zone."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Job:
    name: str
    cmd: list[str]
    cwd: str = "."
    priority: int = 50
    timeout_min: float = 720
    retries: int = 1
    resume_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    report: str | None = None
    notify: bool = True
    heal: bool = True
    max_heals: int = 4
    batch_env: str | None = None
    # filled in by the worker
    status: str = "queued"
    attempts: int = 0
    returncode: int | None = None
    enqueued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    log: str | None = None
    resumed_from: str | None = None
    heal_log: list[dict] = field(default_factory=list)
    incident: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(d) - allowed
        if unknown:
            raise JobError(f"unknown job fields: {sorted(unknown)}")
        job = cls(**d)
        job.validate()
        return job

    def validate(self) -> None:
        if not isinstance(self.name, str) or not _NAME.match(self.name):
            raise JobError(f"bad job name {self.name!r}: use letters, digits, . _ -")
        if not isinstance(self.cmd, list) or not self.cmd or not all(isinstance(c, str) for c in self.cmd):
            raise JobError("cmd must be a non-empty list of strings")
        env_ok = isinstance(self.env, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in self.env.items()
        )
        if not env_ok:
            raise JobError("env must map strings to strings")
        if self.timeout_min <= 0 or self.retries < 0 or self.max_heals < 0:
            raise JobError("timeout_min must be > 0, retries and max_heals >= 0")
        if self.batch_env is not None and (not isinstance(self.batch_env, str) or self.batch_env not in self.env):
            raise JobError("batch_env must name a variable set in env")
        if Path(self.cwd).is_absolute() or ".." in Path(self.cwd).parts[1:]:
            # allow sibling repos like ../scam-radar, nothing deeper outside the lab
            raise JobError("cwd must be relative to LAB_ROOT (one leading .. allowed)")

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_job(path: Path) -> Job:
    return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))


class Queue:
    def __init__(self, paths: LabPaths, notifier: Callable[..., bool] = default_notify,
                 healer: Healer | None = None):
        self.p = paths.ensure()
        self.notify = notifier
        self._healer = healer

    @property
    def healer(self) -> Healer:
        if self._healer is None:
            self._healer = Healer()
        return self._healer

    # ---- queue operations -------------------------------------------------
    def enqueue(self, job: Job | dict) -> Path:
        if isinstance(job, dict):
            job = Job.from_dict(job)
        job.validate()
        job.status, job.enqueued_at = "queued", now_iso()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self.p.queue / f"{job.priority:03d}-{stamp}-{job.name}.json"
        _write_json(path, job.to_dict())
        return path

    def pending(self) -> list[Path]:
        return sorted(self.p.queue.glob("*.json"))

    def running(self) -> list[Path]:
        return sorted(self.p.running.glob("*.json"))

    def recover(self) -> list[Path]:
        """Move jobs interrupted by a crash or power cut back to the front of the queue."""
        moved = []
        for path in self.running():
            target = self.p.queue / ("000-" + path.name.split("-", 1)[1])
            os.replace(path, target)
            moved.append(target)
        return moved

    def claim(self) -> Path | None:
        for path in self.pending():
            target = self.p.running / path.name
            try:
                os.replace(path, target)
            except FileNotFoundError:
                continue  # another worker took it
            return target
        return None

    # ---- running ---------------------------------------------------------
    def run(self, path: Path, runner: Callable[..., subprocess.CompletedProcess] | None = None) -> Job:
        job = _read_job(path)
        cwd = (self.p.root / job.cwd).resolve()
        log_path = self.p.logs / f"{path.stem}.log"
        job.log = str(log_path.relative_to(self.p.root))
        job.started_at = job.started_at or now_iso()
        job.status = "running"
        _write_json(path, job.to_dict())

        run = runner or subprocess.run
        ok = False
        plain_left = job.retries
        while True:
            job.attempts += 1
            env = {**os.environ, **job.env, "LAB_ROOT": str(self.p.root), "LAB_JOB": job.name,
                   "LAB_ATTEMPT": str(job.attempts)}
            if job.resume_dir:
                ckpt = latest_checkpoint((cwd / job.resume_dir).resolve())
                if ckpt is not None:
                    env["LAB_RESUME_FROM"] = str(ckpt)
                    job.resumed_from = str(ckpt)
            cannot_start = timed_out = False
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n=== {now_iso()} attempt {job.attempts} :: {' '.join(job.cmd)}\n")
                log.flush()
                try:
                    result = run(job.cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=job.timeout_min * 60, check=False)
                    job.returncode = result.returncode
                except subprocess.TimeoutExpired:
                    job.returncode = -9
                    timed_out = True
                    log.write(f"=== timed out after {job.timeout_min} min\n")
                except FileNotFoundError as exc:
                    job.returncode = 127
                    cannot_start = True
                    log.write(f"=== cannot start: {exc}\n")
                if not (timed_out or cannot_start):
                    log.write(f"=== exit {job.returncode}{_signame(job.returncode)}\n")
            if job.returncode == 0:
                ok = True
                break
            if cannot_start:
                break
            if not job.heal:
                if job.attempts > job.retries:
                    break
                _write_json(path, job.to_dict())
                continue
            if not self._heal_step(job, path, log_path, cwd, plain_left, timed_out):
                break
            if job.heal_log[-1]["class"] == UNKNOWN:
                plain_left -= 1
            _write_json(path, job.to_dict())

        job.finished_at = now_iso()
        job.status = "done" if ok else "failed"
        dest = (self.p.done if ok else self.p.failed) / path.name
        _write_json(path, job.to_dict())
        os.replace(path, dest)
        if job.notify:
            if ok:
                self.notify(f"done: {job.name}", f"finished in {job.attempts} attempt(s)", tags="white_check_mark")
            else:
                tail = _tail(log_path, 8)
                self.notify(f"FAILED: {job.name}", f"exit {job.returncode} after {job.attempts} attempt(s)\n{tail}",
                            priority="high", tags="x")
        return job

    def _heal_step(self, job: Job, path: Path, log_path: Path, cwd: Path, plain_left: int, timed_out: bool) -> bool:
        """Diagnose the failed attempt, record the decision and apply the fix. True means retry."""
        healer = self.healer
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if timed_out:
            diag = Diagnosis(UNKNOWN, 0.0, "queue", evidence=f"timed out after {job.timeout_min} min")
        else:
            diag = healer.diagnose(text)
        decision = healer.decide(job, diag, plain_left)
        if decision.retry:
            decision = healer.apply(job, decision, self.p, cwd)
        entry = {"attempt": job.attempts, "at": now_iso(), "class": diag.label, "confidence": round(diag.confidence, 3),
                 "source": diag.source, "action": decision.action, "retry": decision.retry,
                 "detail": decision.detail, "evidence": diag.evidence[:200]}
        job.heal_log.append(entry)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"=== heal: {diag.label} ({diag.confidence:.2f}, {diag.source}) -> {decision.action}"
                      f"{': ' + decision.detail if decision.detail else ''}\n")
        if not decision.retry:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            inc = healer.write_incident(job, diag, decision, self.p, _tail(log_path, 60), stamp)
            job.incident = str(inc.relative_to(self.p.root))
            if job.notify:
                self.notify(f"incident: {job.name}", f"{diag.label} ({diag.confidence:.2f}): {decision.detail}\n"
                            f"{job.incident}", priority="high", tags="rotating_light")
        return decision.retry


def _signame(rc: int | None) -> str:
    if rc is None or rc >= 0:
        return ""
    import signal

    try:
        return f" ({signal.Signals(-rc).name})"
    except ValueError:
        return ""


def _tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


@contextmanager
def worker_lock(paths: LabPaths) -> Iterator[None]:
    """Only one worker at a time on this machine."""
    paths.ensure()
    with open(paths.lockfile, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another lab worker is already running") from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def work(
    queue: Queue,
    *,
    once: bool = False,
    deadline: datetime | None = None,
    min_free_gb: float = 50.0,
    idle_sleep: float = 30.0,
    drain: bool = False,
) -> list[Job]:
    """Run jobs until the queue is empty (drain/once), the deadline passes, or forever."""
    finished: list[Job] = []
    with worker_lock(queue.p):
        queue.recover()
        while True:
            if deadline and datetime.now(deadline.tzinfo) >= deadline:
                break
            free = disk_free_gb(queue.p.root)
            if free < min_free_gb:
                queue.notify("disk nearly full", f"{free:.0f} GB free; worker paused", priority="high", tags="warning")
                break
            path = queue.claim()
            if path is None:
                if once or drain:
                    break
                time.sleep(idle_sleep)
                continue
            finished.append(queue.run(path))
            if once:
                break
    return finished
