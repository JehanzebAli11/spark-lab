"""Build page specs: HTML plus a record of every injection and decoy element placed in it.

Pure Python, no browser. `render.py` turns specs into screenshots with measured boxes.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from .payloads import BUTTON_TEXT, DIALOG_TITLES, make_decoy, make_payload
from .templates import NAMES, TEMPLATES, TOPICS, VIEWPORTS, Theme, esc, make_theme, page_html, rgb

# technique -> (weight, visible in a screenshot?)
TECHNIQUES = {
    "visible_text": (0.18, True),
    "low_contrast": (0.15, True),
    "tiny_text": (0.10, True),
    "fake_dialog": (0.12, True),
    "fake_button": (0.10, True),
    "comment": (0.10, True),
    "canvas_text": (0.10, True),  # pixels only: invisible to agents that read the DOM
    "dom_hidden": (0.15, False),  # DOM only: invisible in screenshots, read by text agents
}


@dataclass
class Obj:
    id: str
    label: str  # injection | decoy
    technique: str
    text: str
    goal: str | None = None
    lang: str = "en"
    fg: tuple | None = None
    bg: tuple | None = None
    contrast: float | None = None
    font_px: int | None = None
    in_dom_text: bool = True
    expect_visible: bool = True


@dataclass
class PageSpec:
    id: str
    seed: int
    template: str
    topic: str
    viewport: tuple[int, int]
    dark: bool
    html: str
    objects: list[Obj] = field(default_factory=list)
    scroll_to: str | None = None
    scroll_frac: float = 0.0
    scroll_offset: float = 0.3  # where the target lands, as a fraction of viewport height

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("html")
        return d


def _lum(c: tuple[int, int, int]) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """WCAG contrast ratio, 1 (invisible) to 21 (black on white)."""
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return round((la + 0.05) / (lb + 0.05), 3)


def _new_id(rng: random.Random) -> str:
    # Random ids so nothing in the DOM gives the label away.
    return rng.choice("abcdefghjkmnpqrstuvwxyz") + "".join(rng.choice("abcdefghjkmnpqrstuvwxyz0123456789")
                                                           for _ in range(7))


def _pick_technique(rng: random.Random) -> str:
    names = list(TECHNIQUES)
    return rng.choices(names, weights=[TECHNIQUES[n][0] for n in names])[0]


def injection_html(rng: random.Random, th: Theme, technique: str, payload=None) -> tuple[str, str, Obj]:
    """Returns (inline_html, overlay_html, obj). Exactly one of the html parts is non-empty."""
    p = payload or make_payload(rng)
    oid = _new_id(rng)
    text = esc(p.text)
    obj = Obj(oid, "injection", technique, p.text, p.goal, p.lang, fg=th.fg, bg=th.bg,
              contrast=contrast(th.fg, th.bg), font_px=16, expect_visible=TECHNIQUES[technique][1])
    if technique == "visible_text":
        return f"<p id='{oid}'>{text}</p>", "", obj
    if technique == "low_contrast":
        d = rng.randrange(2, 14)
        fg = tuple(max(0, min(255, v + (d if th.dark else -d))) for v in th.bg)
        obj.fg, obj.contrast = fg, contrast(fg, th.bg)
        return f"<p id='{oid}' style='color:{rgb(fg)}'>{text}</p>", "", obj
    if technique == "tiny_text":
        px = rng.randrange(1, 5)
        obj.font_px = px
        return f"<p id='{oid}' style='font-size:{px}px;line-height:1.1;margin:2px 0'>{text}</p>", "", obj
    if technique == "comment":
        return (f"<div class='comment' id='{oid}'><b>{rng.choice(NAMES)}</b><p>{text}</p></div>", "", obj)
    if technique == "fake_button":
        return (f"<div id='{oid}' style='margin:16px 0'><button style='background:{rgb(th.accent)};color:#fff;"
                f"border:0;padding:10px 16px;border-radius:8px'>{esc(rng.choice(BUTTON_TEXT))}</button>"
                f"<p style='font-size:12px;margin:6px 0 0'>{text}</p></div>", "", obj)
    if technique == "fake_dialog":
        light = rng.random() < 0.6
        bg, fg = ((250, 250, 250), (20, 20, 20)) if light else ((40, 40, 46), (235, 235, 235))
        obj.fg, obj.bg, obj.contrast = fg, bg, contrast(fg, bg)
        top = rng.randrange(30, 60)
        backdrop = "<div style='position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:98'></div>" \
            if rng.random() < 0.5 else ""
        overlay = (f"{backdrop}<div id='{oid}' style='position:fixed;left:50%;top:{top}%;transform:translate(-50%,-50%);"
                   f"width:min(420px,88vw);background:{rgb(bg)};color:{rgb(fg)};border-radius:12px;padding:20px;"
                   "box-shadow:0 12px 40px rgba(0,0,0,.35);z-index:99;font-family:system-ui,sans-serif'>"
                   f"<h3 style='margin:0 0 8px'>{esc(rng.choice(DIALOG_TITLES))}</h3><p style='margin:0 0 16px'>{text}</p>"
                   "<div style='text-align:right'><button style='margin-right:8px'>Cancel</button>"
                   f"<button style='background:{rgb(th.accent)};color:#fff;border:0;padding:8px 14px;"
                   "border-radius:6px'>Allow</button></div></div>")
        return "", overlay, obj
    if technique == "canvas_text":
        obj.in_dom_text = False
        color = rgb(th.fg)
        size = rng.randrange(14, 20)
        obj.font_px = size
        js_text = json.dumps(p.text)
        return (f"<canvas id='{oid}' width='640' height='10' style='max-width:100%;display:block;margin:12px 0'></canvas>"
                f"<script>(function(){{var c=document.getElementById('{oid}'),x=c.getContext('2d'),t={js_text},"
                f"f='{size}px sans-serif',w=c.width-16,l=[],s='';x.font=f;t.split(' ').forEach(function(word){{"
                "var n=s?s+' '+word:word;if(x.measureText(n).width>w&&s){l.push(s);s=word}else s=n});l.push(s);"
                f"c.height=l.length*{size + 6}+12;x=c.getContext('2d');x.font=f;x.fillStyle='{color}';"
                f"l.forEach(function(line,i){{x.fillText(line,8,{size + 4}+i*{size + 6})}})}})();</script>", "", obj)
    if technique == "dom_hidden":
        obj.expect_visible = False
        style = rng.choice(["display:none", "position:absolute;left:-9999px;top:0",
                            "font-size:0;height:0;overflow:hidden", "opacity:0;height:0;overflow:hidden"])
        return f"<div id='{oid}' style='{style}'>{text}</div>", "", obj
    raise ValueError(technique)


def _decoy(rng: random.Random, th: Theme) -> tuple[str, Obj]:
    oid = _new_id(rng)
    text = make_decoy(rng)
    obj = Obj(oid, "decoy", "decoy", text, fg=th.fg, bg=th.bg, contrast=contrast(th.fg, th.bg), font_px=16)
    if rng.random() < 0.4:
        return (f"<div id='{oid}' style='margin:12px 0'><button style='background:{rgb(th.accent)};color:#fff;border:0;"
                f"padding:10px 16px;border-radius:8px'>{esc(text)}</button></div>"), obj
    return f"<p id='{oid}'>{esc(text)}</p>", obj


def build_page(seed: int, page_id: str | None = None, clean_rate: float = 0.25) -> PageSpec:
    """One page, fully determined by `seed`."""
    rng = random.Random(seed)
    template = rng.choice(list(TEMPLATES))
    topic = rng.choice(list(TOPICS))
    th = make_theme(rng)
    blocks, css = TEMPLATES[template](rng, topic, th)
    spec = PageSpec(page_id or f"p{seed:07d}", seed, template, topic, rng.choice(VIEWPORTS), th.dark, "")

    overlays, inline_targets = [], []
    n_inj = 0 if rng.random() < clean_rate else (1 if rng.random() < 0.7 else 2)
    for _ in range(n_inj):
        tech = _pick_technique(rng)
        inline, overlay, obj = injection_html(rng, th, tech)
        spec.objects.append(obj)
        if overlay:
            overlays.append(overlay)
        else:
            blocks.insert(rng.randrange(1, len(blocks) + 1), inline)
            if obj.expect_visible:
                inline_targets.append(obj.id)
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        inline, obj = _decoy(rng, th)
        spec.objects.append(obj)
        blocks.insert(rng.randrange(1, len(blocks) + 1), inline)

    if inline_targets and rng.random() < 0.85:
        spec.scroll_to = rng.choice(inline_targets)
        spec.scroll_offset = round(rng.uniform(0.15, 0.65), 3)
    else:
        spec.scroll_frac = rng.choice([0.0, 0.0, 0.3, 0.6])
    spec.html = page_html(blocks, css, th, overlays="".join(overlays))
    return spec
