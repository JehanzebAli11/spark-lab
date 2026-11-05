import json

import pytest

from guard.screen_injection import TECHNIQUES, build_page, contrast
from guard.screen_injection.__main__ import main as cli
from guard.screen_injection.payloads import make_payload

try:
    import playwright.sync_api  # noqa: F401
    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False
needs_browser = pytest.mark.skipif(not HAVE_PLAYWRIGHT, reason="playwright not installed")


def test_pages_are_deterministic_by_seed():
    a, b = build_page(42), build_page(42)
    assert a.html == b.html and a.to_dict() == b.to_dict()
    assert build_page(43).html != a.html


def test_contrast_ratio():
    assert contrast((0, 0, 0), (255, 255, 255)) == 21.0
    assert contrast((250, 250, 250), (255, 255, 255)) < 1.1


def test_every_technique_appears_and_labels_are_hidden_from_the_dom():
    seen = set()
    for s in range(400):
        spec = build_page(s)
        for o in spec.objects:
            seen.add(o.technique)
            assert o.id in spec.html
        # nothing in the HTML may give the label away
        assert "injection" not in spec.html.lower() and "decoy" not in spec.html.lower()
    assert set(TECHNIQUES) | {"decoy"} == seen


def test_clean_rate_and_low_contrast_is_low():
    specs = [build_page(s) for s in range(300)]
    clean = sum(not any(o.label == "injection" for o in s.objects) for s in specs)
    assert 0.15 < clean / 300 < 0.35
    low = [o for s in specs for o in s.objects if o.technique == "low_contrast"]
    assert low and all(o.contrast < 1.5 for o in low)


def test_payload_params_appear_in_text():
    import random
    rng = random.Random(0)
    for _ in range(200):
        p = make_payload(rng)
        assert all(v in p.text for v in p.params.values())


def test_html_only_cli(tmp_path):
    assert cli(["--n", "5", "--out", str(tmp_path), "--html-only"]) == 0
    specs = [json.loads(x) for x in (tmp_path / "specs.jsonl").read_text().splitlines()]
    assert len(specs) == 5 and len(list((tmp_path / "pages").glob("*.html"))) == 5


@needs_browser
def test_render_measures_boxes_and_resumes(tmp_path):
    assert cli(["--n", "16", "--out", str(tmp_path), "--preview", "2"]) == 0
    recs = [json.loads(x) for x in (tmp_path / "annotations.jsonl").read_text().splitlines()]
    assert len(recs) == 16
    assert len(list((tmp_path / "images").glob("*.png"))) == 16
    assert len(list((tmp_path / "previews").glob("*.png"))) == 2
    for r in recs:
        w, h = r["viewport"]
        for o in r["objects"]:
            if o["on_screen"]:
                x0, y0, x1, y1 = o["box_px"]
                assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
                assert all(0 <= v <= 1 for v in o["box"])
            else:
                assert o["box"] is None
            if o["technique"] == "dom_hidden":
                assert not o["on_screen"] and o["in_dom_text"]
            if o["technique"] == "canvas_text":
                assert not o["in_dom_text"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["pages"] == 16

    # a rerun with more pages renders only the new ones
    assert cli(["--n", "18", "--out", str(tmp_path)]) == 0
    ids = [json.loads(x)["id"] for x in (tmp_path / "annotations.jsonl").read_text().splitlines()]
    assert len(ids) == 18 == len(set(ids))
