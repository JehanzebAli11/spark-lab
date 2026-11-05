"""End-to-end healing: real chaos jobs through the real queue, with a fake clock."""

import json
import sys
from pathlib import Path

import pytest

from lab import report
from lab.chaos import FAULTS
from lab.heal import Healer
from lab.paths import lab_paths
from lab.queue import Job, JobError, Queue, work
from lab.triage import CLASSES, DEFAULT_MODEL, UNKNOWN, LinearClassifier, RuleClassifier, Triage, tail, tokens

REPO = Path(__file__).resolve().parent.parent


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
def waits():
    return []


def chaos_job(name, fault, *extra, **fields):
    env = {"PYTHONPATH": str(REPO), **fields.pop("env", {})}
    extra = extra if "--steps" in extra else ("--steps", "60", *extra)
    cmd = [sys.executable, "-m", "lab.chaos", "--fault", fault, "--seed", "1", *extra]
    return {"name": name, "cmd": cmd, "env": env, "timeout_min": 2, **fields}


def make_queue(lab, waits, triage=None, notes=None):
    healer = Healer(triage or Triage(None), sleep=waits.append)
    return Queue(lab, notifier=notes or Notes(), healer=healer)


def run_one(lab, waits, job, **kw):
    q = make_queue(lab, waits, **kw)
    q.enqueue(job)
    [done] = work(q, drain=True, min_free_gb=0)
    return done


def test_every_fault_has_a_known_label():
    assert {f.label for f in FAULTS.values()} == set(CLASSES)


def test_oom_halves_batch_and_resumes(lab, waits):
    job = run_one(lab, waits, chaos_job("oom", "oom.torch_cuda", "--steps", "400", resume_dir="runs/chaos/oom",
                                        batch_env="BATCH_SIZE", env={"BATCH_SIZE": "64"}))
    assert job.status == "done" and job.attempts == 3
    assert [h["action"] for h in job.heal_log] == ["halve_batch", "halve_batch"]
    assert job.env["BATCH_SIZE"] == "16" and job.env["LAB_BATCH_SCALE"] == "0.25"
    assert job.resumed_from and Path(job.resumed_from).name.startswith("checkpoint-")
    log = (lab.root / job.log).read_text()
    assert "resuming from checkpoint-" in log and "=== heal: oom" in log


def test_disk_full_prunes_runs_then_succeeds(lab, waits):
    run_dir = lab.runs / "chaos" / "disk"
    for step in (100, 200, 300, 400, 500):
        (run_dir / f"checkpoint-{step}").mkdir(parents=True)
    job = run_one(lab, waits, chaos_job("disk", "disk_full.devfull"))
    assert job.status == "done" and job.attempts == 2
    assert job.heal_log[0]["action"] == "prune_retry" and "pruned 3" in job.heal_log[0]["detail"]
    assert not (run_dir / "checkpoint-100").exists() and (run_dir / "checkpoint-500").exists()


def test_network_backs_off_then_recovers(lab, waits):
    job = run_one(lab, waits, chaos_job("net", "network.refused", "--fail-attempts", "2"))
    assert job.status == "done" and job.attempts == 3
    assert waits == [30.0, 60.0]


def test_flaky_retries_straight_away(lab, waits):
    job = run_one(lab, waits, chaos_job("flaky", "flaky.flock"))
    assert job.status == "done" and job.attempts == 2 and waits == []
    assert job.heal_log[0]["class"] == "flaky"


@pytest.mark.parametrize("fault,label", [("code_bug.type", "code_bug"), ("bad_data.jsonl", "bad_data")])
def test_code_and_data_bugs_stop_and_write_incident(lab, waits, fault, label):
    notes = Notes()
    job = run_one(lab, waits, chaos_job("bug", fault, retries=3), notes=notes)
    assert job.status == "failed" and job.attempts == 1  # no wasted retries
    inc = (lab.root / job.incident).read_text()
    assert f"**Class:** {label}" in inc and "**For:** agent" in inc and "## Log tail" in inc
    assert notes.sent[0][0] == "incident: bug"


def test_cuda_error_gets_one_cooldown_then_incident(lab, waits):
    job = run_one(lab, waits, chaos_job("gpu", "cuda_error.launch_failure", "--fail-attempts", "9"))
    assert job.status == "failed" and job.attempts == 2 and waits == [60.0]
    assert [h["action"] for h in job.heal_log] == ["cooldown_retry", "incident"]
    assert "**For:** owner" in (lab.root / job.incident).read_text()


def test_heal_budget_is_respected(lab, waits):
    job = run_one(lab, waits, chaos_job("net", "network.dns", "--fail-attempts", "9", max_heals=1))
    assert job.status == "failed" and job.attempts == 2
    assert job.heal_log[-1]["action"] == "incident" and "budget" in job.heal_log[-1]["detail"]


def test_unknown_uses_plain_retries_then_escalates(lab, waits):
    job = run_one(lab, waits, {"name": "odd", "cmd": [sys.executable, "-c", "import sys; sys.exit(5)"], "retries": 1})
    assert job.status == "failed" and job.attempts == 2
    assert [h["class"] for h in job.heal_log] == [UNKNOWN, UNKNOWN]
    assert [h["action"] for h in job.heal_log] == ["retry", "incident"]


def test_heal_false_keeps_plain_retries(lab, waits):
    job = run_one(lab, waits, chaos_job("raw", "code_bug.type", retries=2, heal=False))
    assert job.attempts == 3 and job.heal_log == [] and job.incident is None


def test_batch_env_must_be_set():
    with pytest.raises(JobError):
        Job.from_dict({"name": "x", "cmd": ["true"], "batch_env": "BATCH_SIZE"})


def test_report_lists_every_decision(lab, waits):
    q = make_queue(lab, waits)
    q.enqueue(chaos_job("net", "network.refused"))
    q.enqueue(chaos_job("bug", "code_bug.name"))
    work(q, drain=True, min_free_gb=0)
    text = report.build_report(lab, gpu="test")
    assert "## Self-healing" in text and "1 recovered, 1 incident(s)" in text
    assert "| net | 1 | network (rules)" in text and "backoff_retry" in text
    assert "| bug | 1 | code_bug (rules)" in text and "incidents/" in text


def test_tail_skips_header_and_old_attempts():
    log = "\n=== t attempt 1 :: python secret --fault oom\nold\n=== exit 1\n\n=== t attempt 2 :: python x\nnew\n"
    assert tail(log) == "new"
    assert "fault" not in tokens(tail(log))


def test_rules_pick_the_last_matching_line():
    text = "retrying after connection reset ... ok\nTypeError: bad operand\n"
    assert RuleClassifier().predict(text).label == "code_bug"


def test_threshold_escalates_low_confidence():
    spec = {"classes": ["code_bug", "oom"], "vocab": {"e:memory": 0}, "idf": [1.0],
            "coef": [[-2.0], [2.0]], "intercept": [0.0, 0.0], "threshold": 0.9}
    tri = Triage(LinearClassifier(spec))
    assert tri.classify("MemoryError").label == "oom"
    assert tri.classify("ValueError: something else").label == UNKNOWN  # 50/50 -> escalate
    assert tri.classify("no traceback here").source == "no-evidence"


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="model not trained yet")
def test_shipped_model_loads_and_classifies_real_chaos_logs(lab, waits):
    tri = Triage.default()
    assert tri.model is not None and 0 < tri.threshold < 1
    job = run_one(lab, waits, chaos_job("oom", "oom.memoryerror", batch_env="BATCH_SIZE",
                                        env={"BATCH_SIZE": "8"}), triage=tri)
    assert job.heal_log[0]["class"] == "oom" and job.heal_log[0]["source"] == "model"
    assert json.loads(DEFAULT_MODEL.read_text())["classes"] == sorted(CLASSES)


def test_triage_server_and_remote_client(tmp_path):
    import threading

    from lab.serve import RemoteTriage, make_server

    srv = make_server("127.0.0.1", 0, Triage(None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        remote = RemoteTriage(url, fallback=Triage(None))
        d = remote.classify("step 1\nOSError: [Errno 28] No space left on device\n=== exit 1")
        assert d.label == "disk_full" and d.source == "remote:rules"
    finally:
        srv.shutdown()
    down = RemoteTriage(url, fallback=Triage(None), timeout=0.5)
    d = down.classify("TypeError: bad\n=== exit 1")
    assert d.label == "code_bug" and "unreachable" in d.source


def test_cli_classify(tmp_path, capsys):
    from lab.cli import main

    p = tmp_path / "x.log"
    p.write_text("Traceback\nsocket.gaierror: [Errno -2] Name or service not known\n=== exit 1\n")
    assert main(["classify", str(p)]) == 0
    assert json.loads(capsys.readouterr().out)["label"] == "network"
