"""Page templates, themes and filler text for the screen-injection generator."""

from __future__ import annotations

import html
import random
from dataclasses import dataclass

TOPICS = {
    "cooking": (["the sauce", "a cast-iron pan", "fresh basil", "the dough", "brown butter", "the stock"],
                ["simmers", "needs", "improves with", "goes well with", "takes", "depends on"],
                ["a pinch of salt", "low heat", "twenty minutes", "patience", "a squeeze of lemon", "good olive oil"]),
    "travel": (["the old town", "the night train", "a window seat", "the coastal road", "the museum pass"],
               ["offers", "saves", "is worth", "fills up with", "rewards"],
               ["early risers", "a full day", "the extra fare", "tourists by noon", "a slower pace"]),
    "tech": (["the new update", "battery life", "the settings menu", "the laptop", "cloud backup"],
             ["improves", "hides", "drains", "protects", "simplifies"],
             ["daily use", "your files", "older devices", "the workflow", "most users"]),
    "finance": (["a savings account", "the monthly budget", "index funds", "credit card fees", "the tax return"],
                ["depends on", "rewards", "eats into", "protects", "simplifies"],
                ["long-term goals", "small habits", "careful planning", "your income", "the interest rate"]),
    "cricket": (["the opening pair", "a short ball", "the spinner", "the new ball", "the run chase"],
                ["needs", "tests", "rewards", "punishes", "changes"],
                ["soft hands", "the batter's footwork", "a patient approach", "the field placement", "early wickets"]),
}
NAMES = ["Ayesha", "Bilal", "Chen", "Dana", "Emeka", "Farah", "Gus", "Hira", "Ivan", "Jules", "Kemi", "Liam"]
PRODUCTS = ["USB-C cable 2m", "Wireless mouse", "Steel water bottle", "Notebook A5", "Desk lamp", "Phone stand",
            "HDMI adapter", "Coffee grinder", "Running socks", "Backpack 20L"]

FONTS = ["Arial, sans-serif", "Georgia, serif", "Verdana, sans-serif", "'Trebuchet MS', sans-serif",
         "'Courier New', monospace", "system-ui, sans-serif"]
VIEWPORTS = [(1280, 800), (1440, 900), (1366, 768), (390, 844), (768, 1024)]


@dataclass
class Theme:
    bg: tuple[int, int, int]
    fg: tuple[int, int, int]
    accent: tuple[int, int, int]
    panel: tuple[int, int, int]
    font: str
    dark: bool


def rgb(c: tuple[int, int, int]) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


def make_theme(rng: random.Random) -> Theme:
    dark = rng.random() < 0.3
    if dark:
        bg = tuple(rng.randrange(10, 40) for _ in range(3))
        fg = tuple(rng.randrange(200, 240) for _ in range(3))
        panel = tuple(min(255, v + 18) for v in bg)
    else:
        bg = tuple(rng.randrange(235, 256) for _ in range(3))
        fg = tuple(rng.randrange(15, 60) for _ in range(3))
        panel = tuple(max(0, v - 12) for v in bg)
    accent = rng.choice([(37, 99, 235), (220, 38, 38), (5, 150, 105), (124, 58, 237), (217, 119, 6)])
    return Theme(bg, fg, accent, panel, rng.choice(FONTS), dark)


def sentence(rng: random.Random, topic: str) -> str:
    subj, verb, obj = TOPICS[topic]
    s = f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)}"
    tail = rng.choice(["", f", according to {rng.choice(NAMES)}", " in most cases", ", which surprised us",
                       " more than you might expect"])
    return s[0].upper() + s[1:] + tail + "."


def paragraph(rng: random.Random, topic: str, n: int | None = None) -> str:
    return " ".join(sentence(rng, topic) for _ in range(n or rng.randrange(2, 5)))


def esc(s: str) -> str:
    return html.escape(s, quote=True)


# Each template returns (body_blocks, layout_css). Blocks are HTML strings; the generator inserts
# injection and decoy elements between them. Block 0 is always the header.

def article(rng, topic, th):
    title = esc(sentence(rng, topic).rstrip("."))
    blocks = [f"<header><h1>{title}</h1><p class='meta'>By {rng.choice(NAMES)} · {rng.randrange(2, 12)} min read</p></header>"]
    for _ in range(rng.randrange(4, 8)):
        if rng.random() < 0.2:
            blocks.append(f"<h2>{esc(sentence(rng, topic).rstrip('.'))}</h2>")
        blocks.append(f"<p>{esc(paragraph(rng, topic))}</p>")
    blocks.append("<section class='comments'><h3>Comments</h3>" + "".join(
        f"<div class='comment'><b>{rng.choice(NAMES)}</b><p>{esc(sentence(rng, topic))}</p></div>"
        for _ in range(rng.randrange(1, 4))) + "</section>")
    return blocks, "main{max-width:720px;margin:24px auto;padding:0 20px} .meta{opacity:.7}"


def shop(rng, topic, th):
    items = rng.sample(PRODUCTS, k=6)
    cards = "".join(
        f"<div class='card'><div class='img'></div><h3>{esc(p)}</h3><p class='price'>${rng.randrange(4, 80)}.{rng.randrange(0, 99):02d}</p>"
        f"<button class='buy'>Add to cart</button></div>" for p in items)
    blocks = [f"<header><h1>Shop · {esc(rng.choice(['Deals', 'New in', 'Bestsellers']))}</h1></header>",
              f"<div class='grid'>{cards}</div>",
              f"<p>{esc(paragraph(rng, 'tech'))}</p>",
              f"<p class='small'>Free returns within 30 days. {esc(sentence(rng, 'finance'))}</p>"]
    css = (".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px}"
           ".card{padding:12px;border-radius:10px;background:var(--panel)} .img{height:90px;border-radius:8px;"
           "background:var(--accent);opacity:.25} .price{font-weight:700} main{max-width:1100px;margin:20px auto;"
           "padding:0 20px} .buy{background:var(--accent);color:#fff;border:0;padding:8px 12px;border-radius:6px}")
    return blocks, css


def inbox(rng, topic, th):
    rows = "".join(
        f"<div class='row'><b>{rng.choice(NAMES)}</b><span>{esc(sentence(rng, rng.choice(list(TOPICS))))}</span>"
        f"<i>{rng.randrange(1, 12)}:{rng.randrange(0, 59):02d}</i></div>" for _ in range(rng.randrange(5, 9)))
    blocks = ["<header><h1>Inbox</h1></header>", f"<div class='list'>{rows}</div>",
              f"<article class='mail'><h2>{esc(sentence(rng, topic).rstrip('.'))}</h2><p>Hi,</p>"
              f"<p>{esc(paragraph(rng, topic))}</p><p>Thanks,<br>{rng.choice(NAMES)}</p></article>"]
    css = (".row{display:flex;gap:12px;padding:8px 10px;border-bottom:1px solid var(--panel)} .row span{flex:1}"
           " main{max-width:960px;margin:16px auto;padding:0 16px} .mail{margin-top:20px;padding:16px;"
           "background:var(--panel);border-radius:10px}")
    return blocks, css


def search(rng, topic, th):
    q = esc(" ".join(rng.choice(TOPICS[topic][0]).split()[-2:]))
    results = "".join(
        f"<div class='res'><a>{esc(sentence(rng, topic).rstrip('.'))}</a><cite>www.{rng.choice(['guide', 'daily', 'notes', 'hub'])}"
        f"{rng.randrange(1, 99)}.example</cite><p>{esc(paragraph(rng, topic, 2))}</p></div>"
        for _ in range(rng.randrange(4, 7)))
    blocks = [f"<header><div class='box'>{q}</div></header>", results]
    css = (".box{padding:10px 16px;border-radius:24px;background:var(--panel);max-width:560px}"
           " .res{margin:18px 0} .res a{color:var(--accent);font-size:18px} cite{display:block;opacity:.7;"
           "font-size:13px} main{max-width:720px;margin:20px 40px}")
    return blocks, css


def docs(rng, topic, th):
    nav = "".join(f"<li>{esc(w.title())}</li>" for w in ["Overview", "Install", "Configure", "API", "FAQ"])
    blocks = ["<header><h1>Documentation</h1></header>", f"<nav><ul>{nav}</ul></nav>",
              f"<h2>{esc(sentence(rng, 'tech').rstrip('.'))}</h2>", f"<p>{esc(paragraph(rng, 'tech'))}</p>",
              f"<pre>pip install example-{rng.randrange(1, 99)}\nexample --help</pre>",
              f"<p>{esc(paragraph(rng, 'tech'))}</p>"]
    css = ("main{max-width:900px;margin:20px auto;padding:0 20px} nav ul{display:flex;gap:16px;list-style:none;"
           "padding:0} pre{background:var(--panel);padding:12px;border-radius:8px}")
    return blocks, css


def social(rng, topic, th):
    posts = "".join(
        f"<div class='post'><b>@{rng.choice(NAMES).lower()}{rng.randrange(1, 999)}</b><p>{esc(paragraph(rng, rng.choice(list(TOPICS)), 2))}</p>"
        f"<span class='acts'>♥ {rng.randrange(0, 900)} · ↻ {rng.randrange(0, 90)}</span></div>"
        for _ in range(rng.randrange(3, 6)))
    blocks = ["<header><h1>Feed</h1></header>", posts]
    css = (".post{padding:14px;margin:12px 0;border-radius:12px;background:var(--panel)} .acts{opacity:.6;"
           "font-size:13px} main{max-width:600px;margin:20px auto;padding:0 16px}")
    return blocks, css


TEMPLATES = {"article": article, "shop": shop, "inbox": inbox, "search": search, "docs": docs, "social": social}


def page_html(blocks: list[str], layout_css: str, th: Theme, overlays: str = "", scripts: str = "") -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f":root{{--accent:{rgb(th.accent)};--panel:{rgb(th.panel)}}}"
        f"body{{margin:0;background:{rgb(th.bg)};color:{rgb(th.fg)};font-family:{th.font};font-size:16px;"
        "line-height:1.5}} h1{font-size:28px} button{font:inherit;cursor:pointer}"
        f"{layout_css}</style></head><body><main>{''.join(blocks)}</main>{overlays}{scripts}</body></html>"
    )
