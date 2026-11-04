import io

import pytest

from lab import teacher
from lab.cli import main
from tests.fake_openai import FakeServer


@pytest.mark.parametrize(
    "text",
    [
        "Chai ke liye doodh aur pani ubaalo, phir patti daalo.",
        "Haan, yeh scam hai kyunke bank kabhi PIN nahi maangta.",
        "Eid Mubarak yaar! Allah aap ko khush rakhe.",
    ],
)
def test_roman_urdu_passes(text):
    ok, reason = teacher.check_roman_urdu(text)
    assert ok, reason


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("چائے بنانے کے لیے پانی اور دودھ ابالیں", "Urdu script"),
        ("Boil water and milk, then add the tea leaves.", "function words"),
    ],
)
def test_roman_urdu_fails(text, why):
    ok, reason = teacher.check_roman_urdu(text)
    assert not ok
    assert why in reason


def test_three_prompts_in_roman_urdu():
    assert len(teacher.PROMPTS) == 3
    for p in teacher.PROMPTS:
        assert teacher.check_roman_urdu(p)[0], p


def test_run_against_fake_server():
    out = io.StringIO()
    with FakeServer() as srv:
        assert teacher.run(srv.url + "/v1", "google/gemma-4-31B-it", out=out) == 0
    assert "3/3 passed" in out.getvalue()
    assert len(srv.requests) == 3
    assert srv.requests[0]["model"] == "google/gemma-4-31B-it"
    assert srv.requests[0]["messages"][0]["role"] == "system"


def test_run_uses_first_model_when_unset():
    out = io.StringIO()
    with FakeServer(model="some/model") as srv:
        assert teacher.run(srv.url + "/v1", None, out=out) == 0
    assert "(some/model)" in out.getvalue()


def test_run_fails_on_english_answers():
    out = io.StringIO()
    with FakeServer(answer="Boil the water first, then add milk.") as srv:
        assert teacher.run(srv.url + "/v1", None, out=out) == 1
    assert "0/3 passed" in out.getvalue()


def test_run_wrong_model_is_unreachable():
    out = io.StringIO()
    with FakeServer(model="other/model") as srv:
        assert teacher.run(srv.url + "/v1", "google/gemma-4-31B-it", out=out) == 2
    assert "not google/gemma-4-31B-it" in out.getvalue()


def test_run_server_down():
    out = io.StringIO()
    assert teacher.run("http://127.0.0.1:9/v1", None, timeout=1, out=out) == 2
    assert "not reachable" in out.getvalue()


def test_cli_teacher_smoke(capsys):
    with FakeServer() as srv:
        assert main(["teacher-smoke", "--base-url", srv.url + "/v1", "--model", "google/gemma-4-31B-it"]) == 0
    assert "3/3 passed" in capsys.readouterr().out
