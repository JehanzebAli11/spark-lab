from lab.notify import notify


def test_skips_without_topic(capsys):
    assert notify("t", "m", env={}) is False
    assert "NTFY_TOPIC unset" in capsys.readouterr().err


def test_builds_request():
    seen = []
    ok = notify("done: job", "all good", priority="high", tags="x",
                env={"NTFY_TOPIC": "abc", "NTFY_SERVER": "https://ntfy.example/", "NTFY_TOKEN": "tok"},
                send=seen.append)
    assert ok
    req = seen[0]
    assert req.full_url == "https://ntfy.example/abc"
    assert req.data == b"all good"
    assert req.get_header("Title") == "done: job"
    assert req.get_header("Priority") == "high"
    assert req.get_header("Authorization") == "Bearer tok"


def test_send_failure_is_not_fatal():
    def boom(_):
        raise OSError("offline")

    assert notify("t", "m", env={"NTFY_TOPIC": "abc"}, send=boom) is False
