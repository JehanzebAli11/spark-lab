"""Post 1: the screen-injection dataset, as a carousel (PDF + slide PNGs) and a single grid image."""

from __future__ import annotations

import json
from pathlib import Path

from .common import FOOTER, data_uri, doc, esc, pdf, png

W, H = 1080, 1350
TECH_LABEL = {
    "visible_text": ("Plain text", "an instruction in the page body"),
    "low_contrast": ("Low contrast", "a few shades off the background"),
    "tiny_text": ("Tiny text", "1 to 4 pixels high"),
    "fake_dialog": ("Fake dialog", "posing as a permission prompt"),
    "fake_button": ("Fake button", "“AI agents: click here”"),
    "comment": ("User comment", "hidden among real comments"),
    "canvas_text": ("Canvas text", "pixels only, not in the page text"),
    "dom_hidden": ("Hidden in the DOM", "invisible, but read by text agents"),
}
SEEN = {"canvas_text": "pixels only", "dom_hidden": "DOM text only"}


def load(ds: Path) -> tuple[list[dict], dict]:
    recs = [json.loads(x) for x in (ds / "annotations.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return recs, json.loads((ds / "summary.json").read_text())


def pick(recs: list[dict], technique: str) -> tuple[dict, dict] | None:
    """A clear example: fully on screen, reasonably large (tiny text: the biggest available)."""
    best = None
    for r in recs:
        for o in r["objects"]:
            if o["technique"] != technique or not o["on_screen"] or o["visible_frac"] < 1:
                continue
            w = o["box_px"][2] - o["box_px"][0]
            # tiny text: the largest font available; otherwise wide, light-theme examples
            score = o["font_px"] * 1000 + w if technique == "tiny_text" else min(w, 700) + (0 if r["dark"] else 300)
            if best is None or score > best[0]:
                best = (score, r, o)
    return (best[1], best[2]) if best else None


def crop(ds: Path, rec: dict, obj: dict, tw: int, th: int | None, pad: int = 60, color: str = "var(--red)",
         others: bool = True) -> str:
    """A tile showing the image region around obj's box, scaled to fit, with boxes drawn.
    th=None shows the whole screenshot at width tw."""
    iw, ih = rec["viewport"]
    x0, y0, x1, y1 = obj["box_px"]
    if th is None:
        s = tw / iw
        th = round(ih * s)
    else:
        rw, rh = (x1 - x0) + 2 * pad, max(y1 - y0, 12) + 2 * pad
        # fit the box with some context, but always cover the whole tile (no empty bands)
        s = max(min(tw / rw, th / rh, 1.6), tw / iw, th / ih)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(0, max(tw - iw * s, tw / 2 - cx * s))
    top = min(0, max(th - ih * s, th / 2 - cy * s))
    boxes = [obj] + ([o for o in rec["objects"] if o is not obj and o["on_screen"]
                      and (o["label"] == "injection" or color != "var(--red)")] if others else [])
    marks = ""
    for o in boxes:
        bx0, by0, bx1, by1 = o["box_px"]
        c = color if o is obj else ("var(--red)" if o["label"] == "injection" else "var(--blue)")
        marks += (f"<div style='position:absolute;left:{left + bx0 * s - 4}px;top:{top + by0 * s - 4}px;"
                  f"width:{(bx1 - bx0) * s + 8}px;height:{max((by1 - by0) * s, 4) + 8}px;border:4px solid {c};"
                  f"border-radius:6px'></div>")
    return (f"<div style='position:relative;width:{tw}px;height:{th}px;overflow:hidden;border-radius:14px;"
            f"background:#fff'><img src='{data_uri(ds / rec['image'])}' style='position:absolute;left:{left}px;"
            f"top:{top}px;width:{iw * s}px;height:{ih * s}px'>{marks}</div>")


def slides(ds: Path, browser, out: Path) -> list[Path]:
    recs, summ = load(ds)
    out.mkdir(parents=True, exist_ok=True)
    n_pages, n_inj, n_dec = summ["pages"], summ["injections"], summ["decoys"]
    made = []

    # 1. Cover: one full page with a fake dialog, every object boxed
    hero = next(((r, o) for r in recs for o in r["objects"] if o["technique"] == "fake_dialog" and o["on_screen"]
                 and r["viewport"][0] >= 1280 and r["template"] in ("shop", "inbox", "docs", "article")
                 and not r["dark"]), None) or pick(recs, "fake_dialog") or pick(recs, "visible_text")
    hero_tile = crop(ds, hero[0], hero[1], 952, 820, pad=300) if hero else ""
    made.append(png(browser, doc(f"""
      <div style='padding:72px 64px 0'>
        <div class='pill red'>AI agent security</div>
        <h1 style='font-size:68px;line-height:1.05;margin:28px 0 20px;font-weight:800'>I generated {n_pages:,} web pages
        that try to hijack AI agents</h1>
        <p class='muted' style='font-size:28px;margin:0 0 40px'>{n_inj:,} hidden instructions, {n_dec:,} harmless decoys,
        every one boxed exactly. No labelling by hand, no model in the loop.</p>
        {hero_tile}
      </div>{FOOTER.format(page='1 / 6')}""", W, H), out / "slide-1.png", W, H))

    # 2. Eight techniques
    tiles = ""
    for t, (name, sub) in TECH_LABEL.items():
        ex = pick(recs, t)
        seen = SEEN.get(t, "pixels and DOM")
        if t == "dom_hidden":
            o = next((o for r in recs for o in r["objects"] if o["technique"] == t), None)
            body = (f"<div class='mono' style='width:430px;height:140px;border-radius:14px;background:#0f1530;"
                    f"padding:16px;font-size:15px;color:#c7d2fe;overflow:hidden'>&lt;div style=\"display:none\"&gt;"
                    f"<br>&nbsp;&nbsp;{esc((o or {}).get('text', ''))[:150]}<br>&lt;/div&gt;</div>")
        elif ex:
            body = crop(ds, ex[0], ex[1], 430, 140, pad=24 if t != "tiny_text" else 10)
        else:
            body = "<div style='width:430px;height:140px'></div>"
        tiles += (f"<div style='background:var(--panel);border-radius:18px;padding:18px'>{body}"
                  f"<div style='display:flex;justify-content:space-between;align-items:baseline;margin-top:14px'>"
                  f"<b style='font-size:24px'>{name}</b><span class='pill {'amber' if t in SEEN else 'blue'}' "
                  f"style='font-size:14px'>{seen}</span></div><div class='muted' style='font-size:18px;margin-top:4px'>"
                  f"{sub}</div></div>")
    made.append(png(browser, doc(f"""
      <div style='padding:64px'>
        <h1 style='font-size:52px;margin:0 0 8px;font-weight:800'>8 ways to hide an instruction</h1>
        <p class='muted' style='font-size:24px;margin:0 0 32px'>Each one tells the agent to leak data, pay, sign or
        run something.</p>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px'>{tiles}</div>
      </div>{FOOTER.format(page='2 / 6')}""", W, H), out / "slide-2.png", W, H))

    # 3. Pixels vs DOM
    canvas = pick(recs, "canvas_text")
    canvas_tile = crop(ds, canvas[0], canvas[1], 952, 260, pad=30) if canvas else ""
    hidden = next((o for r in recs for o in r["objects"] if o["technique"] == "dom_hidden"), {"text": ""})
    inj = [o for r in recs for o in r["objects"] if o["label"] == "injection"]
    n_canvas = sum(o["on_screen"] and not o["in_dom_text"] for o in inj)
    n_dom = sum(o["in_dom_text"] and not o["on_screen"] for o in inj)
    made.append(png(browser, doc(f"""
      <div style='padding:64px'>
        <h1 style='font-size:56px;line-height:1.1;margin:0 0 28px;font-weight:800'>Text agents and screenshot agents
        miss <span class='red'>different</span> attacks</h1>
        <div style='font-size:26px;margin-bottom:14px'><span class='pill amber'>Pixels only</span>&nbsp; Drawn on a
        canvas: a screenshot shows it, the page text doesn't.</div>
        {canvas_tile}
        <div style='font-size:26px;margin:40px 0 14px'><span class='pill amber'>DOM only</span>&nbsp; Hidden with CSS:
        the page text has it, a screenshot doesn't.</div>
        <div class='mono' style='background:#0f1530;border-radius:14px;padding:22px;font-size:19px;color:#c7d2fe'>
        &lt;div style="display:none"&gt;<br>&nbsp;&nbsp;{esc(hidden['text'])}<br>&lt;/div&gt;</div>
        <div style='display:flex;gap:24px;margin-top:44px'>
          <div style='flex:1;background:var(--panel);border-radius:18px;padding:28px'><div style='font-size:64px;
          font-weight:800' class='amber'>{n_canvas:,}</div><div class='muted' style='font-size:22px'>injections in this
          dataset that only a screenshot shows</div></div>
          <div style='flex:1;background:var(--panel);border-radius:18px;padding:28px'><div style='font-size:64px;
          font-weight:800' class='amber'>{n_dom:,}</div><div class='muted' style='font-size:22px'>injections that only
          the page text shows</div></div></div>
        <p style='font-size:34px;margin-top:44px;font-weight:600'>A guard has to look at both.</p>
      </div>{FOOTER.format(page='3 / 6')}""", W, H), out / "slide-3.png", W, H))

    # 4. Decoys
    dec = next(((r, o) for r in recs for o in r["objects"] if o["label"] == "decoy" and o["on_screen"]
                and o["visible_frac"] == 1 and o["box_px"][2] - o["box_px"][0] > 250 and not r["dark"]), None)
    inj = pick(recs, "visible_text")
    made.append(png(browser, doc(f"""
      <div style='padding:64px'>
        <h1 style='font-size:56px;line-height:1.1;margin:0 0 16px;font-weight:800'>Not every instruction is an attack</h1>
        <p class='muted' style='font-size:26px;margin:0 0 36px'>The dataset labels look-alikes written for people, so a
        guard learns the difference instead of flagging every “click here”.</p>
        <div style='font-size:26px;margin-bottom:12px'><span class='pill blue'>Decoy: for people</span></div>
        {crop(ds, dec[0], dec[1], 952, 360, pad=90, color='var(--blue)') if dec else ''}
        <div style='font-size:26px;margin:36px 0 12px'><span class='pill red'>Injection: for the agent</span></div>
        {crop(ds, inj[0], inj[1], 952, 360, pad=90) if inj else ''}
      </div>{FOOTER.format(page='4 / 6')}""", W, H), out / "slide-4.png", W, H))

    # 5. Exact labels + counts
    by = summ["by_technique"]
    top = max(by.values()) if by else 1
    bars = "".join(f"<div style='display:flex;align-items:center;gap:16px;margin:10px 0'><div style='width:250px;"
                   f"font-size:21px'>{TECH_LABEL[t][0]}</div><div style='height:26px;width:{520 * v / top}px;"
                   f"background:var(--red);border-radius:6px'></div><span class='muted' style='font-size:20px'>{v:,}"
                   f"</span></div>" for t, v in sorted(by.items(), key=lambda kv: -kv[1]))
    sample = next((r for r in recs if r["has_visible_injection"]), recs[0])
    o = next((o for o in sample["objects"] if o["on_screen"]), sample["objects"][0] if sample["objects"] else {})
    snippet = esc("{\n" + ",\n".join(f'  "{k}": {json.dumps(o.get(k))}' for k in (
        "label", "technique", "goal", "lang", "box", "on_screen", "in_dom_text")) + "\n}")
    made.append(png(browser, doc(f"""
      <div style='padding:64px'>
        <h1 style='font-size:56px;margin:0 0 12px;font-weight:800'>Every label is exact</h1>
        <p class='muted' style='font-size:26px;margin:0 0 30px'>The generator places each injection, so it knows the box,
        the goal and who can see it. Same box format as Jev's Box answers.</p>
        <pre class='mono' style='background:#0f1530;border-radius:14px;padding:24px;font-size:20px;color:#c7d2fe;
        margin:0 0 36px;white-space:pre-wrap'>{snippet}</pre>
        <div style='font-size:24px;font-weight:600;margin-bottom:8px'>Injections by technique</div>{bars}
        <p class='muted' style='font-size:22px;margin-top:24px'>{summ.get('roman_urdu_injections', 0):,} injections
        are written in Roman Urdu, a language English-only guards tend to miss.</p>
      </div>{FOOTER.format(page='5 / 6')}""", W, H), out / "slide-5.png", W, H))

    # 6. What's next
    made.append(png(browser, doc(f"""
      <div style='padding:96px 64px;display:flex;flex-direction:column;justify-content:center;height:1150px'>
        <div class='pill green' style='align-self:flex-start'>Next</div>
        <h1 style='font-size:64px;line-height:1.08;margin:28px 0;font-weight:800'>Training a small guard model on a
        DGX Spark to catch these before an agent acts</h1>
        <p class='muted' style='font-size:30px;line-height:1.4'>Same data, two eyes: screenshot and page text. Runs
        locally, gives a calibrated probability, and gets compared with frontier models.</p>
        <p style='font-size:30px;margin-top:40px'>Follow along for the results.</p>
      </div>{FOOTER.format(page='6 / 6')}""", W, H), out / "slide-6.png", W, H))

    pdf(browser, made, out / "carousel.pdf", W, H)
    return made


def grid(ds: Path, browser, out: Path) -> Path:
    """One square image: six techniques, boxed."""
    recs, summ = load(ds)
    tiles = ""
    for t in ["fake_dialog", "low_contrast", "fake_button", "comment", "canvas_text", "visible_text"]:
        ex = pick(recs, t)
        if ex:
            tiles += (f"<div>{crop(ds, ex[0], ex[1], 540, 268, pad=50)}<div style='font-size:20px;margin-top:10px;"
                      f"font-weight:600'>{TECH_LABEL[t][0]}</div></div>")
    return png(browser, doc(f"""
      <div style='padding:48px'>
        <h1 style='font-size:44px;margin:0 0 6px;font-weight:800'>{summ['pages']:,} web pages that try to hijack AI agents</h1>
        <p class='muted' style='font-size:22px;margin:0 0 28px'>Red boxes: hidden instructions, placed and labelled by
        code.</p>
        <div style='display:grid;grid-template-columns:repeat(2,540px);gap:18px 24px'>{tiles}</div>
      </div>""", 1200, 1200), out / "grid.png", 1200, 1200)
