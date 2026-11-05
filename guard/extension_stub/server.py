"""python -m guard.extension_stub [--port 8765]

Request (from extension/background.js):
  {url, title, viewport: {w, h}, has_password_field, form_actions: [..], headings: [..],
   screenshot?: base64 png,
   candidates: [{key, tag, hint, text, rect: [x0, y0, x1, y1], attrs: {id, class, src, href},
                 buttons: [{key, text}]}]}
Response:
  {model, verdicts: [{key, kind: ad|cookie_banner|popup|none, p, action: hide|click|none, click_key?}],
   page: {phishing_p, brand, reason}}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

REJECT = re.compile(r"\b(reject( all)?|decline( all)?|refuse|only (necessary|essential)|necessary only|deny)\b", re.I)
BRANDS = {"paypal": "paypal.com", "microsoft": "microsoft.com", "google": "google.com", "apple": "apple.com",
          "metamask": "metamask.io", "coinbase": "coinbase.com", "binance": "binance.com", "amazon": "amazon.com",
          "netflix": "netflix.com", "hbl": "hbl.com", "meezan": "meezanbank.com", "jazzcash": "jazzcash.com.pk"}
MAX_BODY = 20 * 1024 * 1024


_SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu"}


def _registrable(host: str) -> str:
    """Rough registrable domain: example.co.uk, example.com.pk, example.com (no public-suffix list)."""
    parts = host.lower().rstrip(".").split(".")
    if len(parts) > 2 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def fake_verdicts(req: dict, brands: dict | None = None) -> dict:
    """Rule-based verdicts. `brands` maps extra brand names to their real domains (e.g. for demos)."""
    brands = {**BRANDS, **(brands or {})}
    verdicts = []
    for c in req.get("candidates", []):
        hint = c.get("hint")
        ad_iframe = hint == "iframe" and re.search(r"\bads?\b|doubleclick|adservice|adsystem",
                                                    c.get("attrs", {}).get("src", ""), re.I)
        if hint == "ad_hint" or ad_iframe:
            verdicts.append({"key": c["key"], "kind": "ad", "p": 0.9, "action": "hide"})
        elif hint == "cookie_hint":
            btn = next((b for b in c.get("buttons", []) if REJECT.search(b.get("text", ""))), None)
            if btn:
                verdicts.append({"key": c["key"], "kind": "cookie_banner", "p": 0.9, "action": "click",
                                 "click_key": btn["key"]})
            else:
                verdicts.append({"key": c["key"], "kind": "cookie_banner", "p": 0.6, "action": "none"})
        elif hint == "overlay":
            verdicts.append({"key": c["key"], "kind": "popup", "p": 0.55, "action": "none"})
        else:
            verdicts.append({"key": c["key"], "kind": "none", "p": 0.1, "action": "none"})

    host = urlparse(req.get("url", "")).hostname or ""
    shown = " ".join([req.get("title", "")] + req.get("headings", []))
    text = shown.lower()
    brand = next((b for b in brands if b in text), None)
    # the brand as the page spells it ("NovaPay", "PayPal")
    display = shown[text.index(brand):text.index(brand) + len(brand)] if brand else None
    page = {"phishing_p": 0.02, "brand": None, "reason": "no brand or login form"}
    if brand and req.get("has_password_field"):
        real = brands[brand]
        # google.co.uk is Google; google-login.com and paypal.com.evil.test are not.
        if _registrable(host).split(".")[0] != real.split(".")[0]:
            page = {"phishing_p": 0.92, "brand": display,
                    "reason": f"asks for a password as {display} on {host}, not {real}"}
        else:
            page = {"phishing_p": 0.03, "brand": display, "reason": f"real {real} login"}
    return {"model": "stub-rules-0", "verdicts": verdicts, "page": page}


class _Handler(BaseHTTPRequestHandler):
    brands: dict = {}

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # The extension's service worker calls from a chrome-extension:// origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/page-check":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            return self._send(413, {"error": "request too large"})
        try:
            req = json.loads(self.rfile.read(n))
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        return self._send(200, fake_verdicts(req, self.brands))


def make_server(port: int = 8765, host: str = "127.0.0.1", brands: dict | None = None) -> ThreadingHTTPServer:
    handler = type("Handler", (_Handler,), {"brands": dict(brands or {})})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m guard.extension_stub")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--brand", action="append", default=[], metavar="NAME=DOMAIN",
                    help="extra brand and its real domain, e.g. novapay=novapay.example (repeatable)")
    a = ap.parse_args(argv)
    brands = dict(b.lower().split("=", 1) for b in a.brand)
    srv = make_server(a.port, brands=brands)
    print(f"Spark Guard stub listening on http://127.0.0.1:{a.port}/v1/page-check")
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0
