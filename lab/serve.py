"""Serve the failure classifier over HTTP so other repos and agents can use it. Stdlib only.

    python3 -m lab serve-triage --port 8040

    POST /classify   {"log": "<log text>"}  ->  {"label": ..., "confidence": ..., "source": ...,
                                                  "evidence": ..., "scores": {...}, "action": ...}
    GET  /health     {"ok": true, "model": "<meta>", "threshold": ...}

Bind to 127.0.0.1 (the default) or the Tailscale address; there is no authentication.
Set LAB_TRIAGE_URL=http://spark-1:8040 on another machine and `heal.default_triage()` there
(used by the healer) uses the server, falling back to the local model if it does not answer.
"""

from __future__ import annotations

import contextlib
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .heal import ACTIONS
from .triage import UNKNOWN, Diagnosis, Triage

MAX_BODY = 2 << 20


def make_server(host: str = "127.0.0.1", port: int = 8040, triage: Triage | None = None) -> ThreadingHTTPServer:
    tri = triage or Triage.default()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                meta = tri.model.meta if tri.model else {"model": "rules only"}
                return self._send(200, {"ok": True, "model": meta, "threshold": tri.threshold})
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/classify":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413 if n > MAX_BODY else 400, {"error": "body must be 1 byte to 2 MB of JSON"})
            try:
                log = json.loads(self.rfile.read(n))["log"]
                assert isinstance(log, str)
            except (ValueError, KeyError, TypeError, AssertionError):
                return self._send(400, {"error": 'expected {"log": "<text>"}'})
            d = tri.classify(log).to_dict()
            d["action"] = ACTIONS.get(d["label"], "retry, then incident" if d["label"] == UNKNOWN else "incident")
            self._send(200, d)

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


class RemoteTriage(Triage):
    """Classify through a triage server; use the local classifier when it is unreachable."""

    def __init__(self, url: str, fallback: Triage | None = None, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.fallback = fallback or Triage.default()
        super().__init__(self.fallback.model, self.fallback.threshold)
        self.timeout = timeout

    def classify(self, log_text: str) -> Diagnosis:
        req = urllib.request.Request(f"{self.url}/classify", data=json.dumps({"log": log_text}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
            return Diagnosis(d["label"], float(d["confidence"]), "remote:" + d["source"], d.get("evidence", ""),
                             d.get("scores", {}))
        except (OSError, ValueError, KeyError):
            d = self.fallback.classify(log_text)
            d.source += " (remote unreachable)"
            return d


def main(host: str, port: int) -> int:
    srv = make_server(host, port)
    print(f"triage server on http://{host}:{port}  (POST /classify, GET /health)", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0
