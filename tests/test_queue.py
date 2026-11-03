import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

from lab import checkpoints, report
from lab.paths import lab_paths
from lab.queue import Job, JobError, Queue, work, worker_lock


class Notes:
    def __init__(self):
        self.sent = []

    def __call__(self, title, message, **kw):
        self.sent.append((title, message, kw))
        return True


@pytest.fixture
def lab(tmp_path):
    return lab_paths(tmp_path).ensure()


@pytest.fixture
def notes():
    return Notes()


def py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_job_validation_rejects_bad_input():
    with pytest.raises(JobError):
        Job.from_dict({"name": "bad name!", "cmd": ["true"]})
    with pytest.raises(JobError):
        Job.from_dict({"name": "ok", "cmd": "echo hi"})
    with pytest.raises(JobError):
        Job.from_dict({"name": "ok", "cmd": ["true"], "surprise": 1})
    with pytest.raises(JobError):
        Job.from_dict({"name": "ok", "cmd": ["true"], "cwd": "/etc"})
    with pytest.raises(JobError):
        Job.from_dict({"name": "ok", "cmd": ["true"], "cwd": "../../elsewhere"})
    assert Job.from_dict({"name": "ok", "cmd": ["true"], "cwd": "../scam-radar"}).cwd == "../scam-radar"


def test_priority_order_and_success(lab, notes):
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "later", "cmd": py("print('b')"), "priority": 60})
    q.enqueue({"name": "first", "cmd": py("print('a')"), "priority": 10})
    jobs = work(q, drain=True, min_free_gb=0)
    assert [j.name for j in jobs] == ["first", "later"]
    assert all(j.status == "done" for j in jobs)
    assert len(list(lab.done.glob("*.json"))) == 2
    assert "a" in (lab.root / jobs[0].log).read_text()
    assert notes.sent[0][0] == "done: first"


def test_failure_retries_then_fails_and_alerts(lab, notes):
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "boom", "cmd": py("import sys; print('broken'); sys.exit(3)"), "retries": 2})
    [job] = work(q, drain=True, min_free_gb=0)
    assert job.status == "failed" and job.attempts == 3 and job.returncode == 3
    assert list(lab.failed.glob("*.json"))
    title, message, kw = notes.sent[-1]
    assert title == "FAILED: boom" and "broken" in message and kw["priority"] == "high"


def test_retry_succeeds_on_second_attempt(lab, notes):
    flag = lab.root / "flag"
    code = (f"import sys,pathlib; p=pathlib.Path({str(flag)!r}); "
            "sys.exit(0) if p.exists() else (p.touch(), sys.exit(1))")
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "flaky", "cmd": py(code), "retries": 1})
    [job] = work(q, drain=True, min_free_gb=0)
    assert job.status == "done" and job.attempts == 2


def test_resume_from_latest_checkpoint_is_passed(lab, notes):
    run = lab.root / "runs" / "v1"
    for step in (100, 300, 200):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    out = lab.root / "seen.txt"
    code = f"import os,pathlib; pathlib.Path({str(out)!r}).write_text(os.environ.get('LAB_RESUME_FROM',''))"
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "train", "cmd": py(code), "resume_dir": "runs/v1"})
    [job] = work(q, drain=True, min_free_gb=0)
    assert out.read_text().endswith("checkpoint-300")
    assert job.resumed_from.endswith("checkpoint-300")


def test_recover_requeues_interrupted_jobs_first(lab, notes):
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "normal", "cmd": py("pass"), "priority": 10})
    path = q.enqueue({"name": "interrupted", "cmd": py("pass"), "priority": 50})
    path.rename(lab.running / path.name)  # as if the power went out mid-run
    jobs = work(q, drain=True, min_free_gb=0)
    assert [j.name for j in jobs] == ["interrupted", "normal"]


def test_timeout_and_missing_binary(lab, notes):
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "slow", "cmd": py("import time; time.sleep(5)"), "timeout_min": 0.01, "retries": 0})
    q.enqueue({"name": "missing", "cmd": ["definitely-not-a-real-binary-xyz"], "retries": 3})
    jobs = {j.name: j for j in work(q, drain=True, min_free_gb=0)}
    assert jobs["slow"].returncode == -9 and jobs["slow"].status == "failed"
    assert jobs["missing"].returncode == 127 and jobs["missing"].attempts == 1


def test_low_disk_pauses_worker(lab, notes):
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "waits", "cmd": py("pass")})
    assert work(q, drain=True, min_free_gb=10**9) == []
    assert notes.sent[0][0] == "disk nearly full"
    assert len(q.pending()) == 1


def test_deadline_stops_new_jobs(lab, notes):
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "never", "cmd": py("pass")})
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    assert work(q, deadline=past, min_free_gb=0) == []


def test_single_worker_lock(lab):
    with worker_lock(lab), pytest.raises(RuntimeError), worker_lock(lab):
        pass


def test_prune_keeps_newest(tmp_path):
    run = tmp_path / "runs" / "a"
    for s in (10, 20, 30, 40):
        (run / f"checkpoint-{s}").mkdir(parents=True)
    (run / "final").mkdir()
    removed = checkpoints.prune(tmp_path / "runs", keep=2)
    assert sorted(p.name for p in removed) == ["checkpoint-10", "checkpoint-20"]
    assert sorted(p.name for p in run.iterdir()) == ["checkpoint-30", "checkpoint-40", "final"]
    with pytest.raises(ValueError):
        checkpoints.prune(tmp_path, keep=0)


def test_report_lists_jobs_and_quotes_results(lab, notes):
    proj = lab.root / "proj"
    (proj / "out").mkdir(parents=True)
    (proj / "out" / "eval.md").write_text("| metric | value |\n| --- | --- |\n| recall | 0.93 |\n")
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "eval-v1", "cmd": py("pass"), "cwd": "proj", "report": "out/eval.md"})
    q.enqueue({"name": "bad", "cmd": py("import sys; sys.exit(2)"), "retries": 0})
    work(q, drain=True, min_free_gb=0)
    text = report.build_report(lab, report.local_today(), gpu="NVIDIA GB10, 41, 0 %, 1 MiB")
    assert "**1 done, 1 failed, 0 still queued.**" in text
    assert "| recall | 0.93 |" in text
    assert "**bad** exited 2" in text
    assert "NVIDIA GB10" in text
    path = report.write_report(lab, report.local_today(), gpu="x")
    assert path.name == f"{report.local_today().isoformat()}.md"


def test_report_refuses_locked_paths(lab, notes):
    locked = lab.root / "eval" / "locked"
    locked.mkdir(parents=True)
    (locked / "secret.md").write_text("gold labels")
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "sneaky", "cmd": py("pass"), "report": "eval/locked/secret.md"})
    work(q, drain=True, min_free_gb=0)
    text = report.build_report(lab, report.local_today(), gpu="x")
    assert "gold labels" not in text and "refused" in text


def test_cli_enqueue_status_and_worker(lab, tmp_path, capsys):
    from lab.cli import main

    jf = tmp_path / "job.json"
    jf.write_text(json.dumps({"name": "cli-job", "cmd": py("print('hi')"), "notify": False}))
    assert main(["--root", str(lab.root), "enqueue", str(jf)]) == 0
    assert main(["--root", str(lab.root), "status"]) == 0
    assert "cli-job" in capsys.readouterr().out
    assert main(["--root", str(lab.root), "worker", "--drain", "--min-free-gb", "0"]) == 0
    jf.write_text("{not json")
    assert main(["--root", str(lab.root), "enqueue", str(jf)]) == 2


def test_report_window_ignores_old_jobs(lab, notes):
    q = Queue(lab, notifier=notes)
    q.enqueue({"name": "old", "cmd": py("pass")})
    work(q, drain=True, min_free_gb=0)
    future = datetime.now().astimezone() + timedelta(hours=1)
    text = report.build_report(lab, gpu="x", since=future)
    assert "**0 done, 0 failed" in text
