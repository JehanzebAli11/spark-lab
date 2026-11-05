"""VLM doctor client against a fake OpenAI-compatible server (no model in the cloud session)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from labml.doctor import vlm


def test_vlm_client_sends_image_and_parses_answer():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(body)
            reply = '```json\n{"cause": "lr_too_high", "confidence": 0.8, "reason": "spiky loss"}\n```'
            out = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        rows = [{"id": "a", "cause": "lr_too_high", "split": "test", "series": {}},
                {"id": "b", "cause": "healthy", "split": "test", "series": {}}]
        res = vlm.evaluate_rows(rows, f"http://127.0.0.1:{srv.server_address[1]}/v1", "m", lambda s: b"\x89PNG")
    finally:
        srv.shutdown()
    parts = seen["messages"][0]["content"]
    assert seen["model"] == "m" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert [a["pred"] for a in res["answers"]] == ["lr_too_high", "lr_too_high"]
    assert res["summary"]["by_split"]["test"]["accuracy"] == 0.5


def test_parse_falls_back_to_cause_names():
    assert vlm.parse("I think this is overfitting.")["cause"] == "overfitting"
    assert vlm.parse("no idea")["cause"] == "unparsed"
