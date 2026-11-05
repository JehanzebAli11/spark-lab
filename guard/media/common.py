"""Shared look and rendering helpers (Playwright for images, PDFs and video; ffmpeg for MP4)."""

from __future__ import annotations

import base64
import contextlib
import html as _html
import shutil
import subprocess
from pathlib import Path

FONTS = ("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800"
         "&family=JetBrains+Mono:wght@400;600&display=swap'>")
CSS = """
:root{--bg:#0b1020;--panel:#141c36;--panel2:#1b2547;--ink:#eef1fb;--muted:#9aa5c8;--red:#f43f5e;
--blue:#3b82f6;--green:#22c55e;--amber:#f59e0b;--line:#2a3560}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:'JetBrains Mono',ui-monospace,monospace}
.muted{color:var(--muted)} .red{color:var(--red)} .green{color:var(--green)} .blue{color:var(--blue)} .amber{color:var(--amber)}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;font-size:18px}
.pill.red{background:rgba(244,63,94,.15);color:var(--red)} .pill.green{background:rgba(34,197,94,.15);color:var(--green)}
.pill.amber{background:rgba(245,158,11,.15);color:var(--amber)} .pill.blue{background:rgba(59,130,246,.15);color:var(--blue)}
.foot{position:absolute;left:64px;right:64px;bottom:40px;display:flex;justify-content:space-between;color:var(--muted);font-size:20px}
"""
FOOTER = "<div class='foot'><span>spark-lab · built on an NVIDIA DGX Spark</span><span>{page}</span></div>"


def esc(s) -> str:
    return _html.escape(str(s), quote=True)


def doc(body: str, w: int, h: int, css: str = "") -> str:
    return (f"<!doctype html><html><head><meta charset='utf-8'>{FONTS}<style>{CSS}"
            f"body{{width:{w}px;height:{h}px;overflow:hidden;position:relative}}{css}</style></head>"
            f"<body>{body}</body></html>")


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def wait_fonts(page) -> None:
    with contextlib.suppress(Exception):  # offline: the system fonts are fine
        page.evaluate("document.fonts.ready")


def png(browser, html: str, path: Path, w: int, h: int) -> Path:
    page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
    page.set_content(html, wait_until="networkidle")
    wait_fonts(page)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path))
    page.close()
    return path


def pdf(browser, pages: list[str], path: Path, w: int, h: int) -> Path:
    """A LinkedIn document (carousel) post: one PDF page per slide, from slide PNGs."""
    imgs = "".join(f"<img src='{data_uri(p)}' style='display:block;width:{w}px;height:{h}px;page-break-after:always'>"
                   for p in pages)
    page = browser.new_page(viewport={"width": w, "height": h})
    page.set_content(f"<!doctype html><html><head><style>@page{{size:{w}px {h}px;margin:0}}body{{margin:0}}"
                     f"</style></head><body>{imgs}</body></html>")
    page.pdf(path=str(path), width=f"{w}px", height=f"{h}px", print_background=True)
    page.close()
    return path


def to_mp4(webm: Path, mp4: Path, trim: float = 0.0) -> Path:
    """H.264 MP4 for LinkedIn, dropping the first `trim` seconds. Keeps the WebM if ffmpeg is missing."""
    if not shutil.which("ffmpeg"):
        return webm
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{trim:.2f}", "-i", str(webm), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", "-r", "30",
                    str(mp4)], check=True)
    webm.unlink()
    return mp4


def record(browser, html: str, path: Path, w: int, h: int, seconds: float) -> Path:
    """Record an HTML page that animates itself (window.start() runs its timeline) for `seconds`."""
    vdir = path.parent / "_video"
    ctx = browser.new_context(viewport={"width": w, "height": h}, record_video_dir=str(vdir),
                              record_video_size={"width": w, "height": h})
    page = ctx.new_page()
    page.set_content(html, wait_until="networkidle")
    wait_fonts(page)
    page.wait_for_timeout(400)
    t0 = page.evaluate("performance.now()") / 1000
    page.evaluate("window.start && window.start()")
    page.wait_for_timeout(int(seconds * 1000))
    video = page.video
    ctx.close()
    out = to_mp4(Path(video.path()), path.with_suffix(".mp4"), trim=t0)
    shutil.rmtree(vdir, ignore_errors=True)
    return out
