import functools
import http.server
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from guard.extension_stub import fake_verdicts, make_server

EXT = Path(__file__).resolve().parent.parent / "extension"

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False
needs_browser = pytest.mark.skipif(not HAVE_PLAYWRIGHT, reason="playwright not installed")

PAGE = """<!doctype html><html><head><title>PayPal: Log in</title></head><body>
<div id="cookie" style="position:fixed;bottom:0;left:0;right:0;background:#eee;padding:20px">
We use cookies to improve your experience.
<button onclick="window.choice='accept';this.parentNode.remove()">Accept all</button>
<button onclick="window.choice='reject';this.parentNode.remove()">Reject all</button></div>
<div class="ad-slot" style="width:300px;height:250px;background:#fc0">Advertisement</div>
<p class="article">Normal article text that must stay visible.</p>
<h1>Log in to PayPal</h1><form action="/steal"><input type="password" value="hunter2"></form></body></html>"""


def test_stub_cookie_ad_and_phishing():
    out = fake_verdicts({
        "url": "https://paypa1-secure.test/login", "title": "PayPal: Log in", "has_password_field": True,
        "candidates": [
            {"key": "sg1", "hint": "cookie_hint", "buttons": [{"key": "sg2", "text": "Accept all"},
                                                             {"key": "sg3", "text": "Reject all"}]},
            {"key": "sg4", "hint": "ad_hint"},
            {"key": "sg5", "hint": "iframe", "attrs": {"src": "https://video.example/embed"}},
        ]})
    v = {x["key"]: x for x in out["verdicts"]}
    assert v["sg1"]["action"] == "click" and v["sg1"]["click_key"] == "sg3"
    assert v["sg4"]["action"] == "hide"
    assert v["sg5"]["kind"] == "none"  # an iframe is not an ad by itself
    assert out["page"]["phishing_p"] > 0.5


@pytest.mark.parametrize(("url", "title", "phish"), [
    ("https://www.paypal.com/signin", "PayPal", False),
    ("https://accounts.google.co.uk/x", "Sign in - Google", False),
    ("https://www.jazzcash.com.pk/login", "JazzCash login", False),
    ("https://paypal.com.evil.test/x", "PayPal login", True),
    ("https://google-login.com/x", "Google sign in", True),
])
def test_stub_brand_domains(url, title, phish):
    p = fake_verdicts({"url": url, "title": title, "has_password_field": True, "candidates": []})["page"]
    assert (p["phishing_p"] > 0.5) is phish


def test_stub_http_server():
    srv = make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        assert json.loads(opener.open(base + "/health").read())["ok"]
        req = urllib.request.Request(base + "/v1/page-check", data=json.dumps({"candidates": []}).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
        assert json.loads(opener.open(req).read())["page"]["phishing_p"] < 0.5
    finally:
        srv.shutdown()
        srv.server_close()


def test_manifest_is_mv3_and_local_only():
    m = json.loads((EXT / "manifest.json").read_text())
    assert m["manifest_version"] == 3
    bg = (EXT / "background.js").read_text()
    assert "isLocal(s.endpoint)" in bg and "127.0.0.1" in bg


@needs_browser
def test_core_js_finds_candidates_and_applies_verdicts():
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.set_content(PAGE)
        page.add_script_tag(path=str(EXT / "core.js"))
        req = page.evaluate("SparkGuard.buildRequest(document)")
        assert req["has_password_field"] and "hunter2" not in json.dumps(req)  # never reads input values
        hints = {c["hint"] for c in req["candidates"]}
        assert {"cookie_hint", "ad_hint"} <= hints
        done = page.evaluate("(r) => SparkGuard.applyVerdicts(document, r, 0.5)", fake_verdicts(
            dict(req, url="https://paypa1-secure.test/login")))
        assert page.evaluate("window.choice") == "reject"
        assert page.evaluate("getComputedStyle(document.querySelector('.ad-slot')).display") == "none"
        assert page.evaluate("getComputedStyle(document.querySelector('.article')).display") != "none"
        assert page.locator("#spark-guard-warning").count() == 1
        assert {d["did"] for d in done} >= {"warned"}
        b.close()


@needs_browser
def test_extension_end_to_end(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(PAGE)
    web = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(site)))
    stub = make_server(0)
    for s in (web, stub):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            try:
                ctx = p.chromium.launch_persistent_context(
                    str(tmp_path / "profile"), channel="chromium", headless=True,
                    args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
            except Exception as exc:  # noqa: BLE001 - no full Chromium build available here
                pytest.skip(f"cannot launch Chromium with extensions: {exc}")
            sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker", timeout=15000)
            sw.evaluate(f"chrome.storage.local.set({{endpoint: 'http://127.0.0.1:{stub.server_address[1]}/v1/page-check'}})")
            page = ctx.new_page()
            page.goto(f"http://127.0.0.1:{web.server_address[1]}/index.html")
            page.wait_for_selector("#spark-guard-warning", timeout=15000)
            assert page.evaluate("window.choice") == "reject"
            assert page.evaluate("getComputedStyle(document.querySelector('.ad-slot')).display") == "none"
            # a non-local endpoint is refused by the background worker
            sw.evaluate("chrome.storage.local.set({endpoint: 'https://example.com/v1/page-check'})")
            page2 = ctx.new_page()
            page2.goto(f"http://127.0.0.1:{web.server_address[1]}/index.html")
            page2.wait_for_timeout(1500)
            assert page2.locator("#spark-guard-warning").count() == 0
            ctx.close()
    finally:
        for s in (web, stub):
            s.shutdown()
            s.server_close()


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass
