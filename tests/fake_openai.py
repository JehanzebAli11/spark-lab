"""A tiny fake of the vLLM OpenAI API (and the HF whoami endpoint) for tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROMAN_URDU = "Chai banane ke liye pani aur doodh ko ubaalo, phir patti aur cheeni daal kar pakao. Yeh asaan hai."


class FakeServer:
    def __init__(self, model: str = "google/gemma-4-31B-it", answer: str = ROMAN_URDU, token: str = ""):
        self.model, self.answer, self.token = model, answer, token
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [{"id": server.model, "object": "model"}]})
                if self.path == "/api/whoami-v2":
                    if server.token and self.headers.get("Authorization") == f"Bearer {server.token}":
                        return self._json(200, {"name": "tester"})
                    return self._json(401, {"error": "bad token"})
                return self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                if self.path == "/v1/chat/completions":
                    msg = {"role": "assistant", "content": server.answer}
                    return self._json(200, {"choices": [{"index": 0, "message": msg}]})
                return self._json(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
