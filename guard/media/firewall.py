"""Posts 4-5: the action firewall. A chart of attack success by technique and goal, and a replay video
of one poisoned run without and with a guard. Everything is read from firewall run directories."""

from __future__ import annotations

import json
from pathlib import Path

from ..firewall.site import TASK_BY_ID, build_site
from .common import FOOTER, data_uri, doc, esc, png, record

TECH = {"visible_text": "Plain text", "low_contrast": "Low contrast", "tiny_text": "Tiny text",
        "fake_dialog": "Fake dialog", "fake_button": "Fake button", "comment": "User comment",
        "canvas_text": "Canvas text", "dom_hidden": "Hidden in the DOM"}
GOAL = {"exfiltrate": "Leak data", "navigate": "Open a site", "payment": "Send money", "wallet": "Use the wallet",
        "reveal": "Reveal hidden instructions", "execute": "Run a command"}
SERIES_COLORS = ["var(--red)", "var(--blue)", "var(--green)", "var(--amber)"]


def load_run(d: Path) -> tuple[dict, list[dict]]:
    results = json.loads((d / "results.json").read_text())
    rows = [json.loads(x) for x in (d / "actions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return results, rows


def _who(rows: list[dict]) -> tuple[str, str]:
    a = next((r for r in rows if not r.get("summary")), {})
    return a.get("agent", "?"), a.get("guard", "?")


def guard_label(g: str) -> str:
    return {"none": "No guard", "rules": "Rule guard"}.get(g, g.replace("systemone:", "Model: ").replace("llm:", "LLM judge: "))


def chart(runs: list[Path], browser, out: Path, by: str = "technique") -> Path:
    """Grouped horizontal bars: attack success rate per technique (or goal), one bar per guard."""
    series = []
    for d in runs:
        res, rows = load_run(d)
        agent, guard = _who(rows)
        series.append({"label": guard_label(guard), "agent": agent, "data": res[f"attack_success_by_{by}"],
                       "overall": res["attack_success_rate"], "runs": res["poisoned_runs"],
                       "false": res["benign_actions_blocked"]})
    names = TECH if by == "technique" else GOAL
    keys = [k for k in names if any(k in s["data"] for s in series)]
    w, h = 1200, 1350
    row_h = 34 * len(series) + 30
    bar_w = 640
    bars = ""
    for k in keys:
        group = ""
        for i, s in enumerate(series):
            v = s["data"].get(k) or 0
            group += (f"<div style='display:flex;align-items:center;gap:12px;height:30px;margin:4px 0'>"
                      f"<div style='width:{max(2, bar_w * v)}px;height:24px;background:{SERIES_COLORS[i % 4]};"
                      f"border-radius:5px'></div><span style='font-size:19px' class='muted'>{100 * v:.0f}%</span></div>")
        bars += (f"<div style='display:flex;align-items:center;min-height:{row_h}px;border-top:1px solid var(--line)'>"
                 f"<div style='width:330px;font-size:23px'>{esc(names[k])}</div><div>{group}</div></div>")
    legend = "".join(f"<span style='display:inline-flex;align-items:center;gap:10px;margin-right:28px'><span "
                     f"style='width:22px;height:22px;border-radius:5px;background:{SERIES_COLORS[i % 4]}'></span>"
                     f"{esc(s['label'])}: <b>{100 * s['overall']:.0f}%</b> overall</span>" for i, s in enumerate(series))
    agent = series[0]["agent"]
    agent_note = ("a scripted worst-case agent that obeys every instruction it reads (text only)"
                  if agent == "scripted" else f"agent: {agent}")
    title = ("How often a hidden instruction hijacks the agent" if by == "technique"
             else "What each guard lets through, by attack goal")
    body = f"""
      <div style='padding:64px'>
        <div class='pill red'>Attack success rate · lower is better</div>
        <h1 style='font-size:52px;line-height:1.1;margin:24px 0 14px;font-weight:800'>{title}</h1>
        <p class='muted' style='font-size:23px;margin:0 0 26px'>{series[0]['runs']} poisoned runs per guard on 3 tasks,
        {agent_note}. False alarms: {', '.join(f"{esc(s['label'])} {100 * (s['false'] or 0):.0f}%" for s in series)}.</p>
        <div style='font-size:22px;margin-bottom:22px'>{legend}</div>
        {bars}
      </div>{FOOTER.format(page='spark-lab guard.firewall')}"""
    return png(browser, doc(body, w, h), out / f"chart-by-{by}.png", w, h)


def _pick_run(guarded: list[dict], baseline: list[dict]) -> str | None:
    """A poisoned run where the baseline was hijacked and the guard blocked the attack; visible techniques first."""
    hijacked = {r["run_id"] for r in baseline if r.get("summary") and r["attack_executed"]}
    blocked = [r for r in guarded if not r.get("summary") and r["label"] == "attack" and not r["allowed"]
               and r["run_id"] in hijacked]
    order = ["fake_dialog", "fake_button", "visible_text", "comment", "low_contrast", "tiny_text", "dom_hidden"]
    blocked.sort(key=lambda r: order.index(r["technique"]) if r["technique"] in order else 99)
    return blocked[0]["run_id"] if blocked else None


def replay(baseline_dir: Path, guarded_dir: Path, browser, out: Path) -> Path | None:
    """Video: the poisoned page, then the agent's actions without a guard, then with the guard."""
    _, base_rows = load_run(baseline_dir)
    _, guard_rows = load_run(guarded_dir)
    run_id = _pick_run(guard_rows, base_rows)
    if not run_id:
        return None
    first = next(r for r in guard_rows if r["run_id"] == run_id)
    task = TASK_BY_ID[first["task"]]
    seed = int(run_id.split("-")[-2])
    site = build_site(task, seed, first["technique"], first["goal"])
    page = browser.new_page(viewport={"width": 760, "height": 845})
    page.set_content(site.pages[site.injected_url])
    shot = out / f"_page-{run_id}.png"
    out.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(shot))
    page.close()

    def steps(rows):
        out_ = []
        for r in rows:
            if r["run_id"] != run_id or r.get("summary"):
                continue
            a = r["action"]
            args = ", ".join(f"{k}={json.dumps(v)[:60]}" for k, v in a["args"].items())
            out_.append({"call": f"{a['tool']}({args})", "attack": r["label"] == "attack", "allowed": r["allowed"],
                         "reason": r["reason"]})
        return out_

    parts = [{"title": "Without a guard", "steps": steps(base_rows), "guard": guard_label(_who(base_rows)[1])},
             {"title": f"With the {guard_label(_who(guard_rows)[1]).lower()}", "steps": steps(guard_rows),
              "guard": guard_label(_who(guard_rows)[1])}]
    w, h = 1280, 720
    html = doc(f"""
      <div style='display:flex;height:720px'>
        <div style='width:560px;padding:28px;border-right:1px solid var(--line)'>
          <div class='muted' style='font-size:16px;margin-bottom:8px'>The page the agent reads ·
          {esc(TECH.get(first['technique'], first['technique']))}</div>
          <div style='height:560px;overflow:hidden;border-radius:12px'><img src='{data_uri(shot)}'
          style='width:504px'></div>
          <div class='mono' style='font-size:14px;color:#fb7185;margin-top:14px;line-height:1.4'>Hidden instruction:
          {esc(first['injected_text'][:140])}</div>
        </div>
        <div style='flex:1;padding:28px 32px'>
          <div class='muted' style='font-size:16px'>User asked the agent</div>
          <div style='font-size:21px;font-weight:600;margin:6px 0 18px'>{esc(task.request)}</div>
          <div id='title' style='font-size:26px;font-weight:800;margin-bottom:12px'></div>
          <pre id='log' class='mono' style='margin:0;font-size:16px;line-height:1.55;white-space:pre-wrap;
          color:#dbe4ff;height:470px;overflow:hidden'></pre>
        </div>
      </div>
      <script>
      const parts = {json.dumps(parts)};
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      const e = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
      window.start = async () => {{
        const log = document.getElementById('log'), title = document.getElementById('title');
        for (const p of parts) {{
          title.textContent = p.title; log.innerHTML = ''; await sleep(900);
          for (const s of p.steps) {{
            const c = s.attack ? '#fb7185' : '#dbe4ff';
            log.innerHTML += '<span style="color:' + c + '">→ ' + e(s.call) + '</span>\\n'; await sleep(700);
            if (!s.allowed) log.innerHTML += '<span style="color:#f43f5e;font-weight:600">   ✗ BLOCKED: ' + e(s.reason) + '</span>\\n';
            else if (s.attack) log.innerHTML += '<span style="color:#f43f5e;font-weight:600">   ⚠ attack carried out</span>\\n';
            else log.innerHTML += '<span style="color:#22c55e">   ✓ allowed</span>\\n';
            await sleep(650);
          }}
          await sleep(2600);
        }}
      }};
      </script>""", w, h)
    seconds = sum(0.9 + 1.35 * len(p["steps"]) + 2.6 for p in parts) + 1
    path = record(browser, html, out / "firewall-replay", w, h, seconds)
    shot.unlink(missing_ok=True)
    return path
