import os
import shutil
import struct
from pathlib import Path

import pytest

from guard.media.wallet import cases

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False
needs_browser = pytest.mark.skipif(not HAVE_PLAYWRIGHT, reason="playwright not installed")


def png_size(p: Path) -> tuple[int, int]:
    head = p.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_wallet_cases_cover_every_risk_level():
    risks = [c["decoded"].risk for c in cases()]
    assert risks.count("high") == 4 and risks[-1] == "info"
    assert all("USDC of USDC" not in c["decoded"].plain() for c in cases())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    from guard.firewall.__main__ import main as fw
    from guard.screen_injection.__main__ import main as gen
    d = tmp_path_factory.mktemp("media")
    if HAVE_PLAYWRIGHT:
        gen(["--n", "40", "--out", str(d / "ds"), "--preview", "0"])
    for g in ("none", "rules"):
        fw(["--out", str(d / g), "--guard", g, "--per-task", "16"])
    return d


@needs_browser
def test_images_render_at_linkedin_sizes(runs, tmp_path):
    from guard.media import dataset, firewall, wallet
    with sync_playwright() as p:
        b = p.chromium.launch()
        slides = dataset.slides(runs / "ds", b, tmp_path / "p1")
        grid = dataset.grid(runs / "ds", b, tmp_path / "p1")
        wslides = wallet.slides(b, tmp_path / "p2")
        chart = firewall.chart([runs / "none", runs / "rules"], b, tmp_path / "p4")
        b.close()
    assert len(slides) == 6 and all(png_size(s) == (1080, 1350) for s in slides + wslides)
    assert png_size(grid) == (1200, 1200) and png_size(chart) == (1200, 1350)
    assert (tmp_path / "p1" / "carousel.pdf").read_bytes()[:4] == b"%PDF"
    assert (tmp_path / "p2" / "carousel.pdf").exists()


@needs_browser
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_record_makes_an_mp4(tmp_path):
    from guard.media.common import doc, record
    html = doc("<h1 id='x'>0</h1><script>window.start=()=>{let i=0;setInterval(()=>x.textContent=++i,100)}</script>",
               320, 240)
    with sync_playwright() as p:
        b = p.chromium.launch()
        out = record(b, html, tmp_path / "clip", 320, 240, 1.0)
        b.close()
    assert out.suffix == ".mp4" and out.stat().st_size > 1000


@needs_browser
@pytest.mark.skipif(os.environ.get("GUARD_MEDIA_FULL") != "1", reason="slow; set GUARD_MEDIA_FULL=1")
def test_full_videos(runs, tmp_path):
    from guard.media import extension, firewall, wallet
    with sync_playwright() as p:
        b = p.chromium.launch()
        assert wallet.video(b, tmp_path).exists()
        assert firewall.replay(runs / "none", runs / "rules", b, tmp_path).exists()
        b.close()
        assert extension.video(p, tmp_path)["video"].exists()
