"""Demo: a night of chaos jobs through the real worker, animated as a GIF.

    python -m labml.demo queue    # results/demo/queue.gif + the nightly report it produced
    python -m labml.demo doctor   # results/demo/doctor.gif: broken curve -> diagnosis -> fixed curve

Everything shown comes from the run itself (job files, heal_log, report). Back-off waits
are recorded instead of slept so the demo takes seconds.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from lab import report
from lab.heal import Healer
from lab.paths import lab_paths
from lab.queue import Queue, work
from lab.triage import Triage

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "demo"

NIGHT = [
    ("scam-radar-sft", "oom.torch_cuda", {"batch_env": "BATCH_SIZE", "env": {"BATCH_SIZE": "64"},
                                          "resume_dir": "runs/chaos/scam-radar-sft"}, ["--steps", "400"]),
    ("teacher-datagen", "network.refused", {}, ["--fail-attempts", "2"]),
    ("urdu-bench-shards", "disk_full.devfull", {}, []),
    ("hf-cache-warm", "flaky.flock", {}, []),
    ("cricket-pose-train", "cuda_error.illegal_access", {}, ["--fail-attempts", "5"]),
    ("scam-radar-eval", "code_bug.name", {"retries": 2}, []),
    ("sms-import", "bad_data.jsonl", {"retries": 2}, []),
    ("kv-cache-probe", "oom.vllm_kv", {}, []),     # held out when evaluating; the shipped model has seen it
]


def run_night(root: Path) -> tuple[list, str]:
    paths = lab_paths(root).ensure()
    waits: list[float] = []
    q = Queue(paths, notifier=lambda *a, **k: True, healer=Healer(Triage.default(), sleep=waits.append))
    for step in (100, 200, 300, 400, 500):
        (paths.runs / "chaos" / "urdu-bench-shards" / f"checkpoint-{step}").mkdir(parents=True)
    for i, (name, fault, fields, extra) in enumerate(NIGHT):
        env = {"PYTHONPATH": str(REPO), **fields.pop("env", {})}
        q.enqueue({"name": name, "priority": 10 + i, "timeout_min": 2, "env": env, **fields,
                   "cmd": [sys.executable, "-m", "lab.chaos", "--fault", fault, "--seed", str(i), *extra]})
    jobs = work(q, drain=True, min_free_gb=0)
    return jobs, report.build_report(paths, gpu="(demo: CPU container)")


def _font(size: int):
    from PIL import ImageFont

    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


COLORS = {"done": (80, 200, 120), "failed": (235, 90, 80), "running": (240, 200, 80), "queued": (150, 150, 160)}


def queue_gif(jobs, out: Path) -> None:
    from PIL import Image, ImageDraw

    font, big = _font(15), _font(19)
    w, h = 1060, 120 + 34 * len(jobs) + 60
    state = {j.name: ("queued", "", "") for j in jobs}
    frames = []

    def frame(caption: str, hold: int = 1):
        im = Image.new("RGB", (w, h), (24, 26, 33))
        d = ImageDraw.Draw(im)
        d.text((24, 18), "spark-lab worker  ·  self-healing queue  ·  chaos night", font=big, fill=(230, 230, 240))
        d.text((24, 52), f"{'job':22s}{'status':9s}{'diagnosis (confidence)':30s}action", font=font,
               fill=(140, 150, 170))
        for i, j in enumerate(jobs):
            st, diag, act = state[j.name]
            y = 84 + 34 * i
            d.text((24, y), f"{j.name:22s}", font=font, fill=(220, 220, 230))
            d.text((24 + 9 * 22, y), f"{st:9s}", font=font, fill=COLORS[st if st in COLORS else "running"])
            d.text((24 + 9 * 31, y), f"{diag:30s}", font=font, fill=(200, 200, 210))
            d.text((24 + 9 * 61, y), act[:52], font=font, fill=(170, 200, 255))
        d.text((24, h - 44), caption[:110], font=font, fill=(255, 220, 140))
        frames.extend([im] * hold)

    frame("8 jobs queued for tonight", 3)
    for j in jobs:
        for hstep in j.heal_log:
            state[j.name] = ("running", f"{hstep['class']} ({hstep['confidence']:.2f})",
                             f"{hstep['action']}: {hstep['detail']}" if hstep["detail"] else hstep["action"])
            frame(f"{j.name}: attempt {hstep['attempt']} failed -> {hstep['class']} -> {hstep['action']}", 2)
        st = j.status
        last = state[j.name]
        tail = f"done after {j.attempts} attempt(s)" if st == "done" else f"incident: {j.incident}"
        state[j.name] = (st, last[1] or "-", last[2] if last[2] else tail)
        frame(f"{j.name}: {tail}", 2)
    fixed = sum(1 for j in jobs if j.status == "done")
    frame(f"night over: {fixed}/{len(jobs)} jobs finished, {len(jobs) - fixed} incident file(s) written", 6)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=700, loop=0, optimize=True)


def doctor_gif(out: Path) -> None:
    """One harmed test run per cause: broken curve, the doctor's call, the curve after the fix."""
    import io
    import json

    from PIL import Image, ImageDraw

    from .doctor.render import png_bytes

    path = REPO / "results" / "doctor" / "autofix_runs.jsonl"
    runs_path = REPO / "datasets" / "doctor-runs" / "runs.jsonl.gz"
    if not path.exists():
        raise SystemExit("run `python -m labml.doctor.autofix` first")
    loops = [json.loads(line) for line in path.read_text().splitlines()]
    import gzip

    from .doctor.dataset import _clean
    from .doctor.sim import RunConfig, apply_fix, run

    with gzip.open(runs_path, "rt") as fh:
        rows = {r["id"]: r for r in map(json.loads, fh)}
    font, big = _font(15), _font(18)
    frames = []
    seen = set()
    for lp in loops:
        if lp["policy"] != "ensemble" or not lp["rescued"] or lp["cause"] in seen or lp["split"] != "test":
            continue
        seen.add(lp["cause"])
        row = rows[lp["id"]]
        cfg = RunConfig(**row["config"])
        for s in lp["steps"]:
            if s["fix"] not in ("stop", "escalate"):
                cfg = apply_fix(cfg, s["fix"])
        fixed = _clean(run(cfg))
        before = Image.open(io.BytesIO(png_bytes(row["series"], 320, 240))).convert("RGB")
        after = Image.open(io.BytesIO(png_bytes(fixed["series"], 320, 240))).convert("RGB")
        diag = ", ".join(f"{s['diagnosis']} ({s.get('confidence', 0):.2f})" for s in lp["steps"]
                         if s["fix"] not in ("stop", "escalate"))
        for show_after in (False, True):
            im = Image.new("RGB", (720, 360), (250, 250, 252))
            d = ImageDraw.Draw(im)
            d.text((20, 14), f"injected: {lp['cause']}", font=big, fill=(20, 20, 30))
            im.paste(before, (20, 60))
            d.text((20, 306), f"before: test acc {lp['before']['test_acc'] or 0:.2f}", font=font, fill=(160, 40, 40))
            d.text((20, 328), f"doctor: {diag}", font=font, fill=(30, 60, 160))
            if show_after:
                im.paste(after, (380, 60))
                d.text((380, 306), f"after fix: test acc {lp['after']['test_acc'] or 0:.2f}", font=font,
                       fill=(30, 130, 60))
                d.text((380, 328), f"reference {lp['reference']['test_acc']:.2f}", font=font, fill=(90, 90, 90))
            frames += [im] * (2 if not show_after else 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=900, loop=0, optimize=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="labml.demo")
    ap.add_argument("what", choices=["queue", "doctor"])
    a = ap.parse_args(argv)
    if a.what == "queue":
        root = Path(tempfile.mkdtemp(prefix="chaos-night-"))
        try:
            jobs, text = run_night(root)
            OUT.mkdir(parents=True, exist_ok=True)
            text = text.replace(str(root), "~/lab/spark-lab")
            (OUT / "queue_report.md").write_text(text, encoding="utf-8")
            inc = sorted((root / "incidents").glob("*.md"))
            if inc:
                (OUT / "example_incident.md").write_text(inc[0].read_text().replace(str(REPO), "~/lab/spark-lab")
                                                         .replace(str(root), "~/lab/spark-lab"), encoding="utf-8")
            queue_gif(jobs, OUT / "queue.gif")
            print(text)
        finally:
            shutil.rmtree(root, ignore_errors=True)
    else:
        doctor_gif(OUT / "doctor.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
