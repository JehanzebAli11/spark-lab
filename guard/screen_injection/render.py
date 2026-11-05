"""Render page specs in headless Chromium and measure every element's box on the screenshot.

Needs Playwright (`pip install playwright && python -m playwright install chromium`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .generate import PageSpec

_MEASURE = """
(ids) => {
  const vw = window.innerWidth, vh = window.innerHeight, out = {};
  for (const id of ids) {
    const el = document.getElementById(id);
    if (!el) { out[id] = null; continue; }
    // Text elements: hug the rendered text, not the full-width block. Canvas and dialogs: the element.
    let r = el.getBoundingClientRect();
    if (el.tagName !== 'CANVAS' && getComputedStyle(el).position !== 'fixed') {
      const range = document.createRange();
      range.selectNodeContents(el);
      const t = range.getBoundingClientRect();
      if (t.width > 0 && t.height > 0) r = t;
    }
    const st = getComputedStyle(el);
    const x0 = Math.max(0, r.left), y0 = Math.max(0, r.top);
    const x1 = Math.min(vw, r.right), y1 = Math.min(vh, r.bottom);
    const full = Math.max(1, r.width * r.height);
    const shown = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    const hidden = st.display === 'none' || st.visibility === 'hidden' || parseFloat(st.opacity) === 0;
    out[id] = {box: [x0, y0, x1, y1], visible_frac: hidden ? 0 : shown / full, area: shown};
  }
  return out;
}
"""

_SCROLL = """
([id, frac, offset]) => {
  if (id) {
    const el = document.getElementById(id);
    const r = el.getBoundingClientRect();
    window.scrollTo(0, Math.max(0, window.scrollY + r.top - window.innerHeight * offset));
  } else {
    window.scrollTo(0, (document.body.scrollHeight - window.innerHeight) * frac);
  }
}
"""

_PREVIEW = """
(boxes) => {
  for (const b of boxes) {
    const d = document.createElement('div');
    const c = b.label === 'injection' ? '#e11d48' : '#2563eb';
    Object.assign(d.style, {position: 'fixed', left: b.box[0] + 'px', top: b.box[1] + 'px',
      width: (b.box[2] - b.box[0]) + 'px', height: (b.box[3] - b.box[1]) + 'px',
      border: '3px solid ' + c, zIndex: 2147483647, pointerEvents: 'none', boxSizing: 'border-box'});
    const t = document.createElement('div');
    t.textContent = b.label === 'injection' ? 'injection: ' + b.technique : 'decoy (benign)';
    Object.assign(t.style, {position: 'absolute', left: '-3px', top: '-22px', background: c, color: '#fff',
      font: '600 12px system-ui, sans-serif', padding: '2px 6px', whiteSpace: 'nowrap'});
    d.appendChild(t);
    document.body.appendChild(d);
  }
}
"""

MIN_VISIBLE = 0.5  # an object counts as on-screen when at least half of it is in the viewport


def render(specs: Iterable[PageSpec], out: Path, preview: int = 0) -> list[dict]:
    """Render each spec, appending one line per page to annotations.jsonl as it goes (crash-safe)."""
    from playwright.sync_api import sync_playwright  # optional dependency

    out = Path(out)
    for d in ("images", "pages", "previews"):
        (out / d).mkdir(parents=True, exist_ok=True)
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for i, spec in enumerate(specs):
            w, h = spec.viewport
            page.set_viewport_size({"width": w, "height": h})
            page.set_content(spec.html, wait_until="load")
            page.evaluate(_SCROLL, [spec.scroll_to, spec.scroll_frac, spec.scroll_offset])
            measured = page.evaluate(_MEASURE, [o.id for o in spec.objects])
            img = out / "images" / f"{spec.id}.png"
            page.screenshot(path=str(img))
            (out / "pages" / f"{spec.id}.html").write_text(spec.html, encoding="utf-8")

            objects = []
            for o in spec.objects:
                m = measured.get(o.id) or {"box": [0, 0, 0, 0], "visible_frac": 0, "area": 0}
                on_screen = o.expect_visible and m["visible_frac"] >= MIN_VISIBLE and m["area"] >= 16
                x0, y0, x1, y1 = (round(v, 1) for v in m["box"])
                rec = {
                    "label": o.label, "technique": o.technique, "text": o.text, "goal": o.goal, "lang": o.lang,
                    "on_screen": on_screen, "in_dom_text": o.in_dom_text,
                    "box_px": [x0, y0, x1, y1] if on_screen else None,
                    # Same convention as Jev's Box answers: edges as fractions of width and height.
                    "box": [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)]
                    if on_screen else None,
                    "visible_frac": round(m["visible_frac"], 3), "contrast": o.contrast, "font_px": o.font_px,
                }
                objects.append(rec)
            rec = {
                "id": spec.id, "image": f"images/{spec.id}.png", "html": f"pages/{spec.id}.html", "seed": spec.seed,
                "template": spec.template, "topic": spec.topic, "viewport": [w, h], "dark": spec.dark,
                "has_visible_injection": any(o["label"] == "injection" and o["on_screen"] for o in objects),
                "has_dom_injection": any(o["label"] == "injection" and o["in_dom_text"] for o in objects),
                "objects": objects,
            }
            records.append(rec)
            with open(out / "annotations.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if i < preview:
                boxes = [dict(o, box=o["box_px"]) for o in objects if o["on_screen"]]
                page.evaluate(_PREVIEW, boxes)
                page.screenshot(path=str(out / "previews" / f"{spec.id}.png"))
        browser.close()
    return records
