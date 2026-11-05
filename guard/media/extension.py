"""Post 3: the Chrome extension, recorded for real in Chromium with the extension loaded.

The news site and the 'NovaPay' brand are fictional, so nothing imitates a real company.
"""

from __future__ import annotations

import functools
import http.server
import shutil
import tempfile
import threading
from pathlib import Path

from ..extension_stub import make_server
from .common import FONTS, to_mp4

EXT = Path(__file__).resolve().parents[2] / "extension"

NEWS = """<!doctype html><html><head><meta charset='utf-8'><title>The Daily Byte</title>""" + FONTS + """<style>
body{margin:0;font-family:Georgia,serif;background:#fafaf7;color:#1c1c1c}
header{padding:18px 40px;border-bottom:1px solid #ddd;display:flex;justify-content:space-between;align-items:center}
header b{font-size:30px;font-family:Inter,sans-serif;letter-spacing:-.5px}
.wrap{display:flex;gap:32px;max-width:1180px;margin:0 auto;padding:24px 40px}
article{flex:1;font-size:19px;line-height:1.6} h1{font-size:40px;line-height:1.15;margin:8px 0 16px}
aside{width:300px}
.ad-banner{height:110px;background:linear-gradient(90deg,#ffb703,#fb8500);color:#fff;font:800 30px Inter,sans-serif;
display:flex;align-items:center;justify-content:center;margin:0 40px 0;border-radius:8px}
.sponsored{height:260px;background:linear-gradient(160deg,#8338ec,#3a86ff);border-radius:10px;color:#fff;padding:20px;
font:700 24px Inter,sans-serif;margin-bottom:18px}
.sponsored small{display:block;font:500 13px Inter,sans-serif;opacity:.8;margin-bottom:8px}
#cookie{position:fixed;left:0;right:0;bottom:0;background:#fff;border-top:3px solid #1c1c1c;padding:22px 40px;
display:flex;gap:16px;align-items:center;font:16px Inter,sans-serif;box-shadow:0 -10px 30px rgba(0,0,0,.15)}
#cookie p{flex:1;margin:0} #cookie button{font:600 16px Inter,sans-serif;padding:12px 18px;border-radius:8px;border:0}
.accept{background:#1c1c1c;color:#fff} .reject{background:#eee}
</style></head><body>
<header><b>The Daily Byte</b><span style='font-family:Inter,sans-serif;color:#666'>Tech · Science · Money</span></header>
<div class='ad-banner' data-ad>MEGA SALE — 70% OFF EVERYTHING</div>
<div class='wrap'><article><div style='font-family:Inter,sans-serif;color:#b45309;font-weight:600'>ANALYSIS</div>
<h1>Why small models are quietly taking over everyday AI</h1>
<p>For years the story of AI was bigger models in bigger data centres. The newest twist runs the other way: small
models, trained for one job, running on the device in front of you.</p>
<p>They answer in milliseconds, cost almost nothing per call, and keep private data at home. For checks that happen
thousands of times a day, such as spam, safety and routing, that matters more than raw intelligence.</p>
<p>The catch is trust. A small model has to know when it is unsure, and hand off when it is.</p></article>
<aside><div class='sponsored'><small>Sponsored</small>Earn 12% a month with CryptoMax. Limited spots!</div>
<div class='sponsored' style='background:linear-gradient(160deg,#06d6a0,#118ab2)'><small>Promoted</small>
Try SnackBox free for 30 days</div></aside></div>
<div id='cookie'><p>We and our 812 partners use cookies to personalise ads and track your visits.</p>
<button class='reject' onclick="this.parentNode.remove()">Reject all</button>
<button class='accept' onclick="this.parentNode.remove()">Accept all</button></div>
</body></html>"""

LOGIN = """<!doctype html><html><head><meta charset='utf-8'><title>NovaPay: Log in</title>""" + FONTS + """<style>
body{margin:0;font-family:Inter,sans-serif;background:#f3f6fb;display:flex;align-items:center;justify-content:center;
height:100vh} .card{background:#fff;width:400px;padding:40px;border-radius:16px;box-shadow:0 20px 50px rgba(20,40,90,.12)}
.logo{font:800 30px Inter,sans-serif;color:#1d4ed8;margin-bottom:24px} label{display:block;font-size:14px;color:#555;
margin:14px 0 6px} input{width:100%;box-sizing:border-box;padding:12px;font-size:16px;border:1px solid #cbd5e1;
border-radius:8px} button{width:100%;margin-top:22px;padding:14px;border:0;border-radius:8px;background:#1d4ed8;
color:#fff;font:600 17px Inter,sans-serif}
</style></head><body><div class='card'><div class='logo'>NovaPay</div><h1 style='font-size:22px;margin:0 0 6px'>
Log in to NovaPay</h1><p style='color:#666;margin:0 0 10px'>Your account has been limited. Verify to restore access.</p>
<form action='/collect'><label>Email</label><input id='email' type='email'><label>Password</label>
<input type='password'><button type='button'>Log in</button></form></div></body></html>"""

CAPTION_JS = """([t, where]) => {
  let c = document.getElementById('sg-caption');
  if (!c) {
    c = document.createElement('div');
    c.id = 'sg-caption';
    c.dataset.sgHidden = 'caption';  // the extension skips its own demo captions
    Object.assign(c.style, {position: 'fixed', left: '50%', top: '18px', transform: 'translateX(-50%)',
      zIndex: 2147483646, background: 'rgba(11,16,32,.92)', color: '#fff', font: '600 22px Inter, sans-serif',
      padding: '14px 24px', borderRadius: '12px', maxWidth: '1100px', textAlign: 'center',
      boxShadow: '0 10px 30px rgba(0,0,0,.25)'});
    document.body.appendChild(c);
  }
  c.style.top = where === 'bottom' ? 'auto' : '18px';
  c.style.bottom = where === 'bottom' ? '24px' : 'auto';
  c.textContent = t;
}"""

END = """<!doctype html><html><head><meta charset='utf-8'>""" + FONTS + """<style>body{margin:0;height:100vh;
background:#0b1020;color:#eef1fb;font-family:Inter,sans-serif;display:flex;flex-direction:column;align-items:center;
justify-content:center} h1{font-size:54px;margin:0 0 14px} p{font-size:24px;color:#9aa5c8;margin:6px}</style></head>
<body><h1>Spark Guard</h1><p>A small model on your own machine, not someone else's cloud.</p>
<p>Ads and cookie walls handled · phishing flagged · no page data leaves the laptop.</p>
<p style='margin-top:28px;font-size:20px'>Built on an NVIDIA DGX Spark · spark-lab</p></body></html>"""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def video(pw, out: Path) -> dict:
    """Records extension-demo.mp4 and saves before/after/phishing stills. Needs the full Chromium build."""
    out.mkdir(parents=True, exist_ok=True)
    site = Path(tempfile.mkdtemp())
    (site / "index.html").write_text(NEWS, encoding="utf-8")
    (site / "login").mkdir()
    (site / "login" / "index.html").write_text(LOGIN, encoding="utf-8")
    web = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(site)))
    stub = make_server(0, brands={"novapay": "novapay.example"})
    for s in (web, stub):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    port, live = web.server_address[1], f"http://127.0.0.1:{stub.server_address[1]}/v1/page-check"
    vdir = out / "_video"
    profile = tempfile.mkdtemp()
    try:
        ctx = pw.chromium.launch_persistent_context(
            profile, channel="chromium", headless=True, viewport={"width": 1280, "height": 720},
            record_video_dir=str(vdir), record_video_size={"width": 1280, "height": 720},
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}",
                  "--host-resolver-rules=MAP *.test 127.0.0.1"])
        sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker", timeout=15000)
        sw.evaluate("chrome.storage.local.set({endpoint: 'http://127.0.0.1:9/v1/page-check'})")  # guard off
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        def caption(t: str, where: str = "top") -> None:
            page.evaluate(CAPTION_JS, [t, where])

        page.goto(f"http://dailybyte.test:{port}/")
        caption("A normal news site: banner ad, sponsored boxes and a cookie wall")
        page.wait_for_timeout(3500)
        page.screenshot(path=str(out / "before.png"))
        sw.evaluate(f"chrome.storage.local.set({{endpoint: '{live}'}})")  # guard on
        caption("Spark Guard on: a small model on this machine decides what to hide…")
        page.wait_for_function("!document.getElementById('cookie')", timeout=15000)
        page.wait_for_timeout(600)
        caption("Ads hidden · cookies rejected · nothing left the laptop")
        page.wait_for_timeout(3200)
        page.screenshot(path=str(out / "after.png"))

        page.goto(f"http://novapay-verify.test:{port}/login/")
        caption("A NovaPay login page… on novapay-verify.test, not novapay.example", "bottom")
        page.wait_for_selector("#spark-guard-warning", timeout=15000)
        page.wait_for_timeout(1200)
        page.type("#email", "me@example.com", delay=80)
        page.wait_for_timeout(2800)
        page.screenshot(path=str(out / "phishing.png"))
        page.set_content(END)
        page.wait_for_timeout(3500)
        video = page.video
        ctx.close()
        mp4 = to_mp4(Path(video.path()), out / "extension-demo.mp4", trim=0.4)
    finally:
        for s in (web, stub):
            s.shutdown()
            s.server_close()
        shutil.rmtree(vdir, ignore_errors=True)
        shutil.rmtree(site, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)
    return {"video": mp4, "before": out / "before.png", "after": out / "after.png", "phishing": out / "phishing.png"}
