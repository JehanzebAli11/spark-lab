"""scripts/bootstrap.sh, run end to end against stub system commands and a fake teacher.

The stubs record every call, so the tests can check that the script never runs sudo, writes the
env file, pulls each image once, and does nothing new on a second run.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tests.fake_openai import FakeServer

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "bootstrap.sh"
TOKEN = "hf_" + "a1B2c3D4e5" * 3
VERIFIED_IMAGE = (
    "vllm/vllm-openai:gemma4-cu130@sha256:0d152595cd940ea1e0890fa190a37b7a86e3f7f0f5048ac3e2e4c14529fea833"
)

STUBS = {
    "sudo": 'echo "$*" >>"$STATE/sudo.log"; exit 1',
    "id": '[ "${1:-}" = -u ] && { echo 1000; exit 0; }; exec /usr/bin/id "$@"',
    "uname": "echo aarch64",
    "tmux": "exit 0",
    "loginctl": "echo yes",
    "nvidia-smi": 'case "$*" in *driver_version*) echo 580.95.05 ;; *) echo "NVIDIA GB10" ;; esac',
    "getent": '[ "$1 $2" = "passwd agent" ] && { echo "agent:x:1001:1001::/home/agent:/bin/bash"; exit 0; }; exit 2',
    "docker": r"""
echo "$*" >>"$STATE/docker.log"
key="$STATE/img-$(printf '%s' "${3:-$2}" | md5sum | cut -c1-12)"
case "$1" in
  info|run|rm|stop) exit 0 ;;
  logs) echo "INFO fake vllm log line" ;;
  image) [ -f "$key" ] ;;
  pull) touch "$STATE/img-$(printf '%s' "$2" | md5sum | cut -c1-12)" ;;
  *) exit 0 ;;
esac
""",
    "systemctl": r"""
echo "$*" >>"$STATE/systemctl.log"
[ "${1:-}" = --user ] && shift
case "$1 ${2:-}" in
  "is-enabled sleep.target")
    n=$(( $(cat "$STATE/sleep_checks" 2>/dev/null || echo 0) + 1 )); echo $n >"$STATE/sleep_checks"
    if [ "$n" -gt "${SLEEP_MASKED_AFTER:-0}" ]; then echo masked; else echo enabled; fi ;;
  show*) echo 0 ;;
  is-failed*) exit 1 ;;
  is-active*) exit 1 ;;
  *) exit 0 ;;
esac
""",
}


@pytest.fixture
def env(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash missing")
    fakebin, state, home, lab_root, sshd = (tmp_path / d for d in ("bin", "state", "home", "labroot", "ssh"))
    for d in (fakebin, state, home, lab_root, sshd / "sshd_config.d"):
        d.mkdir(parents=True)
    (sshd / "sshd_config").write_text("PasswordAuthentication no\n")
    for name, body in STUBS.items():
        f = fakebin / name
        f.write_text("#!/usr/bin/env bash\n" + body.lstrip() + "\n")
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    e = {k: v for k, v in os.environ.items() if not k.startswith(("NTFY_", "HF_", "TEACHER_", "XDG_"))}
    e.update(
        PATH=f"{fakebin}:{os.environ['PATH']}",
        HOME=str(home),
        USER="tester",
        STATE=str(state),
        LAB_ROOT=str(lab_root),
        SSHD_CONFIG_DIR=str(sshd),
        MODEL_MIN_FREE_GB="0",
        TEACHER_WAIT_MIN="1",
        BOOTSTRAP_POLL_SEC="1",
        BOOTSTRAP_JOB_WAIT_SEC="0",
        BOOTSTRAP_NOTIFY="0",
        no_proxy="127.0.0.1,localhost",
        NO_PROXY="127.0.0.1,localhost",
    )
    return e


def run(env: dict, stdin: str = "", **extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT)], input=stdin, env={**env, **extra}, capture_output=True, text=True, timeout=120
    )


def lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def read_env(env: dict) -> dict[str, str]:
    out = {}
    for line in (Path(env["HOME"]) / ".config/spark-lab.env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_full_run_then_rerun_is_idempotent(env):
    state, home = Path(env["STATE"]), Path(env["HOME"])
    with FakeServer(token=TOKEN) as srv:
        env.update(TEACHER_URL=srv.url + "/v1", HF_ENDPOINT=srv.url)

        first = run(env, stdin=f"my-test-topic-123\n{TOKEN}\n")
        out = first.stdout + first.stderr
        assert first.returncode == 0, out
        assert "NTFY topic" in out
        assert "HF token works (account: tester)" in out
        assert "teacher serving google/gemma-4-31B-it" in out
        assert "teacher smoke: 3/3 passed" in out
        assert "smoke job queued" in out
        assert len(srv.requests) == 3

        cfg = read_env(env)
        assert cfg["NTFY_TOPIC"] == "my-test-topic-123"
        assert cfg["HF_TOKEN"] == TOKEN
        assert cfg["VLLM_IMAGE"] == VERIFIED_IMAGE
        assert cfg["TEACHER_MODEL"] == "google/gemma-4-31B-it"
        assert stat.S_IMODE((home / ".config/spark-lab.env").stat().st_mode) == 0o600
        assert TOKEN not in out  # the token is never echoed

        units = home / ".config/systemd/user"
        assert {p.name for p in units.iterdir()} >= {"vllm-teacher.service", "lab-worker.service", "lab-nightly.timer"}
        sysctl = lines(state / "systemctl.log")
        for u in ("lab-nightly.timer", "lab-worker.service", "vllm-teacher.service"):
            assert f"--user enable --now {u}" in sysctl
        docker = lines(state / "docker.log")
        assert [d for d in docker if d.startswith("pull")] == [
            f"pull {VERIFIED_IMAGE}",
            "pull nvcr.io/nvidia/pytorch:25.11-py3",
        ]
        download = [d for d in docker if d.startswith("run --rm --entrypoint python3")]
        assert len(download) == 1 and VERIFIED_IMAGE in download[0] and "--user 1000:" in download[0]
        queued = list((Path(env["LAB_ROOT"]) / "jobs/queue").glob("*-smoke-python.json"))
        assert len(queued) == 1

        env_before = (home / ".config/spark-lab.env").read_text()
        (state / "docker.log").unlink()
        (state / "systemctl.log").unlink()

        second = run(env, stdin="")  # no input available: nothing may be asked
        out2 = second.stdout + second.stderr
        assert second.returncode == 0, out2
        assert "NTFY topic" not in out2 and "HF read token" not in out2
        assert "NTFY_TOPIC already set" in out2 and "HF_TOKEN already set" in out2
        assert "smoke job already in jobs/queue/" in out2
        assert (home / ".config/spark-lab.env").read_text() == env_before
        assert not [d for d in lines(state / "docker.log") if d.startswith("pull")]
        assert "--user daemon-reload" not in lines(state / "systemctl.log")
        assert not any(" restart " in f" {s} " for s in lines(state / "systemctl.log"))
        assert len(list((Path(env["LAB_ROOT"]) / "jobs/queue").glob("*-smoke-python.json"))) == 1

    assert not (state / "sudo.log").exists(), "bootstrap must never run sudo"


def test_sudo_step_prints_exact_line_and_waits(env):
    """sleep.target is unmasked on the first check and masked after the owner presses Enter."""
    state = Path(env["STATE"])
    with FakeServer(token=TOKEN) as srv:
        env.update(TEACHER_URL=srv.url + "/v1", HF_ENDPOINT=srv.url)
        res = run(env, stdin=f"\nmy-test-topic-123\n{TOKEN}\n", SLEEP_MASKED_AFTER="1")
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target" in out
    assert "Press Enter when done to re-check" in out
    assert "ok    Spark never sleeps" in out
    assert not (state / "sudo.log").exists()


def test_skipped_sudo_step_lands_in_checklist(env):
    state = Path(env["STATE"])
    with FakeServer(token=TOKEN) as srv:
        env.update(TEACHER_URL=srv.url + "/v1", HF_ENDPOINT=srv.url)
        res = run(env, stdin=f"\nskip\nmy-test-topic-123\n{TOKEN}\n", SLEEP_MASKED_AFTER="99")
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "not done yet" in out
    assert "[ ] Spark never sleeps (skipped; run: sudo systemctl mask sleep.target" in out
    assert not (state / "sudo.log").exists()


def test_failing_teacher_fails_the_run(env):
    with FakeServer(answer="Boil the water, then add milk.", token=TOKEN) as srv:
        env.update(TEACHER_URL=srv.url + "/v1", HF_ENDPOINT=srv.url)
        res = run(env, stdin=f"my-test-topic-123\n{TOKEN}\n")
    assert res.returncode == 1
    assert "FAILED: teacher smoke test failed" in res.stdout


def test_teacher_never_ready_times_out(env):
    with FakeServer(model="other/model", token=TOKEN) as srv:
        env.update(TEACHER_URL=srv.url + "/v1", HF_ENDPOINT=srv.url, TEACHER_WAIT_MIN="0")
        res = run(env, stdin=f"my-test-topic-123\n{TOKEN}\n")
    assert res.returncode == 1
    assert "teacher not ready after 0 min" in res.stdout


def test_refuses_root(env):
    stub = Path(env["PATH"].split(":")[0]) / "id"
    stub.write_text('#!/usr/bin/env bash\n[ "${1:-}" = -u ] && { echo 0; exit 0; }; exec /usr/bin/id "$@"\n')
    res = run(env)
    assert res.returncode == 1
    assert "not as root" in res.stderr


def test_keeps_custom_image_and_fixes_old_model_case(env):
    cfg = Path(env["HOME"]) / ".config/spark-lab.env"
    cfg.parent.mkdir(parents=True)
    example = (REPO / "systemd/spark-lab.env.example").read_text()
    example = example.replace(VERIFIED_IMAGE, "nvcr.io/nvidia/vllm:26.10-py3")
    example = example.replace("TEACHER_MODEL=google/gemma-4-31B-it", "TEACHER_MODEL=google/gemma-4-31b-it")
    example = example.replace("NTFY_TOPIC=spark-lab-CHANGE-ME-to-something-random", "NTFY_TOPIC=already-chosen-1")
    example = example.replace("HF_TOKEN=", f"HF_TOKEN={TOKEN}")
    cfg.write_text(example)
    with FakeServer(token=TOKEN) as srv:
        env.update(TEACHER_URL=srv.url + "/v1", HF_ENDPOINT=srv.url)
        res = run(env)
    assert res.returncode == 0, res.stdout + res.stderr
    got = read_env(env)
    assert got["VLLM_IMAGE"] == "nvcr.io/nvidia/vllm:26.10-py3"
    assert got["TEACHER_MODEL"] == "google/gemma-4-31B-it"
    assert got["NTFY_TOPIC"] == "already-chosen-1"
    assert "kept your choice" in res.stdout


def _sourced(env: dict, code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {code}'], env=env, capture_output=True, text=True, timeout=30
    )


@pytest.mark.parametrize(
    ("topic", "ok"),
    [("spark-lab-0123abcd", True), ("short", False), ("has space here", False),
     ("spark-lab-CHANGE-ME-to-something-random", False), ("a/b/c/d/e/f", False)],
)
def test_valid_topic(env, topic, ok):
    assert (_sourced(env, f"valid_topic '{topic}'").returncode == 0) is ok


@pytest.mark.parametrize(("token", "ok"), [(TOKEN, True), ("hf_short", False), ("sk-" + "a" * 30, False), ("", False)])
def test_valid_hf_token(env, token, ok):
    assert (_sourced(env, f"valid_hf_token '{token}'").returncode == 0) is ok


def test_env_set_replaces_and_appends(env, tmp_path):
    f = tmp_path / "x.env"
    f.write_text("# comment\nA=1\nAB=keep\nA=2\n")
    res = _sourced({**env, "SPARK_LAB_ENV": str(f)}, "env_set A 'x&y|z'; env_set NEW v; env_get A; env_get AB")
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ["x&y|z", "keep"]
    assert f.read_text() == "# comment\nA=x&y|z\nAB=keep\nNEW=v\n"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600


def test_env_get_strips_quotes(env, tmp_path):
    f = tmp_path / "x.env"
    f.write_text('Q="a b c"\nS=\'d e\'\n')
    res = _sourced({**env, "SPARK_LAB_ENV": str(f)}, "env_get Q; env_get S; env_get MISSING")
    assert res.stdout.splitlines() == ["a b c", "d e"]


def test_shellcheck_clean():
    sc = shutil.which("shellcheck")
    if sc is None:
        pytest.skip("shellcheck not installed (pip install shellcheck-py)")
    scripts = [SCRIPT, REPO / "scripts/setup_spark.sh", REPO / "jobs/nightly.sh", REPO / "publish/publish.sh"]
    res = subprocess.run([sc, *map(str, scripts)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout
