"""Write the nightly markdown report: what ran, what failed, results, machine health."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

from .paths import LabPaths

MAX_QUOTED_LINES = 80


def _jobs(folder: Path, since: datetime) -> list[dict]:
    """Jobs in folder that finished at or after `since` (timezone-aware)."""
    out = []
    for p in sorted(folder.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            finished = datetime.fromisoformat(d["finished_at"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if finished >= since:
            out.append(d)
    return out


def _minutes(job: dict) -> str:
    try:
        a = datetime.fromisoformat(job["started_at"])
        b = datetime.fromisoformat(job["finished_at"])
        return f"{(b - a).total_seconds() / 60:.0f}"
    except (KeyError, TypeError, ValueError):
        return "?"


def gpu_summary() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or out.stderr.strip() or "no output"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({exc})"


def local_today() -> date:
    return datetime.now().astimezone().date()


def build_report(
    paths: LabPaths,
    day: date | None = None,
    gpu: str | None = None,
    since: datetime | None = None,
) -> str:
    """Report on jobs finished since `since` (default: the last 24 hours), titled by local date."""
    day_s = (day or local_today()).isoformat()
    since = since or datetime.now().astimezone() - timedelta(hours=24)
    done = _jobs(paths.done, since)
    failed = _jobs(paths.failed, since)
    pending = sorted(paths.queue.glob("*.json"))
    usage = shutil.disk_usage(paths.root)

    lines = [f"# Nightly report {day_s}", ""]
    lines.append(f"**{len(done)} done, {len(failed)} failed, {len(pending)} still queued.**")
    lines += ["", "## Jobs", ""]
    if done or failed:
        lines += ["| Job | Status | Attempts | Minutes | Resumed from | Log |", "| --- | --- | --- | --- | --- | --- |"]
        for j in failed + done:
            lines.append(
                f"| {j['name']} | {j['status']} | {j.get('attempts', '?')} | {_minutes(j)} | "
                f"{Path(j['resumed_from']).name if j.get('resumed_from') else '-'} | `{j.get('log') or '-'}` |")
    else:
        lines.append("No jobs finished in the last 24 hours.")

    quoted = [j for j in done if j.get("report")]
    if quoted:
        lines += ["", "## Results", ""]
        for j in quoted:
            rp = (paths.root / j.get("cwd", ".") / j["report"]).resolve()
            lines += [f"### {j['name']}", ""]
            if "eval/locked" in rp.as_posix():
                lines.append("_refused: report path is inside eval/locked_")
            elif rp.is_file():
                text = rp.read_text(encoding="utf-8", errors="replace").splitlines()
                lines += text[:MAX_QUOTED_LINES]
                if len(text) > MAX_QUOTED_LINES:
                    lines.append(f"_…{len(text) - MAX_QUOTED_LINES} more lines in `{rp.name}`_")
            else:
                lines.append(f"_expected report not found: `{j['report']}`_")
            lines.append("")

    healed = [j for j in failed + done if j.get("heal_log")]
    if healed:
        n_fixed = sum(1 for j in done if j.get("heal_log"))
        n_inc = sum(1 for j in failed + done if j.get("incident"))
        lines += ["", "## Self-healing", "",
                  f"{len(healed)} job(s) failed at least once: {n_fixed} recovered, {n_inc} incident(s) written.", "",
                  "| Job | Attempt | Diagnosis | Confidence | Action | Detail |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for j in healed:
            for h in j["heal_log"]:
                detail = (h.get("detail") or "-").replace("|", "/")
                lines.append(f"| {j['name']} | {h.get('attempt')} | {h.get('class')} ({h.get('source', '?')}) | "
                             f"{h.get('confidence', 0):.2f} | {h.get('action')} | {detail} |")
        incidents = [j for j in failed + done if j.get("incident")]
        if incidents:
            lines += ["", "Incidents:", ""]
            lines += [f"- `{j['incident']}` ({j['heal_log'][-1].get('class')})" for j in incidents]

    if failed:
        lines += ["", "## Failures", ""]
        for j in failed:
            lines.append(f"- **{j['name']}** exited {j.get('returncode')} after {j.get('attempts')} attempt(s); "
                         f"see `{j.get('log')}`")

    lines += ["", "## Machine", "",
              f"- Disk: {usage.free / 1e9:.0f} GB free of {usage.total / 1e9:.0f} GB",
              f"- GPU: {gpu if gpu is not None else gpu_summary()}",
              f"- Next in queue: {', '.join(p.stem.split('-', 2)[-1] for p in pending[:5]) or 'nothing'}", ""]
    return "\n".join(lines)


def write_report(
    paths: LabPaths, day: date | None = None, gpu: str | None = None, since: datetime | None = None
) -> Path:
    paths.ensure()
    day = day or local_today()
    text = build_report(paths, day, gpu, since)
    out = paths.reports / f"{day.isoformat()}.md"
    out.write_text(text, encoding="utf-8")
    return out
