"""Command line: python -m lab <command>."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

from . import checkpoints, report
from .notify import notify
from .paths import lab_paths
from .queue import JobError, Queue, disk_free_gb, work


def _deadline(hhmm: str | None) -> datetime | None:
    if not hhmm:
        return None
    now = datetime.now().astimezone()
    h, m = (int(x) for x in hhmm.split(":"))
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        from datetime import timedelta

        target += timedelta(days=1)
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lab", description="Spark job queue and nightly reports")
    ap.add_argument("--root", help="lab root (default LAB_ROOT or repo root)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enqueue", help="add job JSON file(s) to the queue")
    e.add_argument("files", nargs="+", type=Path)

    w = sub.add_parser("worker", help="run queued jobs")
    w.add_argument("--once", action="store_true", help="run one job then exit")
    w.add_argument("--drain", action="store_true", help="exit when the queue is empty")
    w.add_argument("--until", metavar="HH:MM", help="stop starting new jobs after this local time")
    w.add_argument("--min-free-gb", type=float, default=50.0)

    sub.add_parser("status", help="show queue, running job and disk")

    r = sub.add_parser("report", help="write reports/<date>.md")
    r.add_argument("--date", default="today")

    pr = sub.add_parser("prune", help="keep only the newest N checkpoints per run")
    pr.add_argument("root_dir", type=Path)
    pr.add_argument("--keep", type=int, default=3)
    pr.add_argument("--dry-run", action="store_true")

    n = sub.add_parser("notify", help="send a test notification")
    n.add_argument("message", nargs="?", default="hello from the Spark")

    a = ap.parse_args(argv)
    paths = lab_paths(a.root)

    if a.cmd == "enqueue":
        q = Queue(paths)
        for f in a.files:
            try:
                path = q.enqueue(json.loads(f.read_text(encoding="utf-8")))
            except (JobError, json.JSONDecodeError, TypeError) as exc:
                print(f"{f}: {exc}", file=sys.stderr)
                return 2
            print(f"queued {path.name}")
        return 0

    if a.cmd == "worker":
        jobs = work(Queue(paths), once=a.once, drain=a.drain, deadline=_deadline(a.until), min_free_gb=a.min_free_gb)
        failed = [j for j in jobs if j.status == "failed"]
        print(f"{len(jobs)} job(s) run, {len(failed)} failed")
        return 1 if failed else 0

    if a.cmd == "status":
        q = Queue(paths)
        print(f"queued:  {len(q.pending())}")
        for p in q.pending()[:10]:
            print(f"  {p.name}")
        print(f"running: {', '.join(p.name for p in q.running()) or '-'}")
        print(f"disk free: {disk_free_gb(paths.root):.0f} GB")
        print(f"gpu: {report.gpu_summary()}")
        return 0

    if a.cmd == "report":
        day = report.local_today() if a.date == "today" else date.fromisoformat(a.date)
        print(report.write_report(paths, day))
        return 0

    if a.cmd == "prune":
        removed = checkpoints.prune(a.root_dir, keep=a.keep, dry_run=a.dry_run)
        for p in removed:
            print(("would remove " if a.dry_run else "removed ") + str(p))
        return 0

    if a.cmd == "notify":
        return 0 if notify("spark-lab test", a.message) else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
