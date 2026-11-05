"""Post 2: the wallet decoder. Carousel of 'what you sign vs what it means', plus a terminal demo video."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ..wallet import Names, explain_transaction, explain_typed_data
from ..wallet.abi import encode
from ..wallet.keccak import selector
from .common import FOOTER, doc, esc, pdf, png, record

W, H = 1080, 1350
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # the real USDC contract, a public address
SPENDER = "0x7a3F9bE0c1D25e4F8aB6C3d9E1f2A4b5C6d7E8f9"  # an arbitrary unknown address
NFT = "0x5E1f2D3c4B5a69788796A5b4C3d2E1f0A9B8c7D6"   # an arbitrary NFT collection
ME = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
FRIEND = "0x2222222222222222222222222222222222222222"
NOW = 1_800_000_000
NAMES = {"tokens": {USDC: {"symbol": "USDC", "decimals": 6}}, "known": {FRIEND: "Sara (contact)"}}


def _call(sig, types, values):
    return selector(sig) + encode(types, values).hex()


def cases() -> list[dict]:
    names = Names(NAMES["tokens"], NAMES["known"])
    seaport = {"domain": {"name": "Seaport", "version": "1.6", "chainId": 1}, "primaryType": "OrderComponents",
               "message": {"offerer": ME, "offer": [{"itemType": 2, "token": NFT, "identifierOrCriteria": "4417",
                                                      "startAmount": "1", "endAmount": "1"}],
                           "consideration": [{"itemType": 0, "token": "0x" + "0" * 40, "identifierOrCriteria": "0",
                                              "startAmount": "1", "endAmount": "1", "recipient": SPENDER}]}}
    fake_permit2 = {"domain": {"name": "Permit2", "chainId": 1, "verifyingContract": SPENDER},
                    "primaryType": "PermitSingle",
                    "message": {"details": {"token": USDC, "amount": str(2**160 - 1), "expiration": "0", "nonce": "0"},
                                "spender": SPENDER, "sigDeadline": str(NOW + 1800)}}
    tx1 = {"to": USDC, "data": _call("approve(address,uint256)", ["address", "uint256"], [SPENDER, 2**256 - 1])}
    tx2 = {"to": NFT, "data": _call("setApprovalForAll(address,bool)", ["address", "bool"], [SPENDER, True])}
    tx3 = {"to": USDC, "data": _call("transfer(address,uint256)", ["address", "uint256"], [FRIEND, 40_000_000])}
    return [
        {"title": "“Claim your free airdrop”", "kind": "Transaction", "raw": tx1["data"],
         "decoded": explain_transaction(tx1, names), "cli": ["tx", "--to", USDC, "--data", tx1["data"]]},
        {"title": "“Verify your NFT to join the mint”", "kind": "Transaction", "raw": tx2["data"],
         "decoded": explain_transaction(tx2, names), "cli": ["tx", "--to", NFT, "--data", tx2["data"]]},
        {"title": "“Sign in to the marketplace”", "kind": "Signature (EIP-712)",
         "raw": json.dumps(seaport, indent=1), "decoded": explain_typed_data(seaport, names, now=NOW),
         "typed": seaport},
        {"title": "“Gasless approval, just sign”", "kind": "Signature (EIP-712)",
         "raw": json.dumps(fake_permit2, indent=1), "decoded": explain_typed_data(fake_permit2, names, now=NOW),
         "typed": fake_permit2},
        {"title": "Paying a friend", "kind": "Transaction", "raw": tx3["data"],
         "decoded": explain_transaction(tx3, names), "cli": ["tx", "--to", USDC, "--data", tx3["data"]]},
    ]


RISK = {"high": ("red", "HIGH RISK"), "medium": ("amber", "CHECK"), "info": ("green", "LOOKS FINE")}


def _case_slide(c: dict, i: int, n: int) -> str:
    d = c["decoded"]
    color, word = RISK[d.risk]
    raw = c["raw"]
    raw = (raw[:420] + "…") if len(raw) > 420 else raw
    effects = "".join(f"<p style='font-size:38px;line-height:1.3;margin:0 0 16px;font-weight:600'>"
                      f"{esc(e.summary.strip(' -'))}</p>" for e in d.effects)
    flags = "".join(f"<li style='margin:8px 0'>{esc(f.message)}</li>" for f in d.flags if f.severity != "info")
    return doc(f"""
      <div style='padding:0 64px;height:1220px;display:flex;flex-direction:column;justify-content:center'>
        <div class='muted' style='font-size:26px'>The site says</div>
        <h1 style='font-size:60px;margin:6px 0 40px;font-weight:800'>{c['title']}</h1>
        <div class='muted' style='font-size:22px;margin-bottom:10px'>What your wallet is asked to sign · {c['kind']}</div>
        <pre class='mono' style='background:#0f1530;border-radius:14px;padding:26px;font-size:21px;line-height:1.45;color:#93a3d8;
        white-space:pre-wrap;word-break:break-all;margin:0 0 44px;max-height:420px;overflow:hidden'>{esc(raw)}</pre>
        <div style='display:flex;align-items:center;gap:16px;margin-bottom:18px'>
          <div class='muted' style='font-size:22px'>What it actually does</div><span class='pill {color}'>{word}</span></div>
        <div style='background:var(--panel);border-radius:18px;padding:36px;border-left:8px solid var(--{color})'>
          {effects}
          {f"<ul class='muted' style='font-size:28px;margin:16px 0 0;padding-left:28px'>{flags}</ul>" if flags else ""}
        </div>
      </div>{FOOTER.format(page=f'{i} / {n}')}""", W, H)


def slides(browser, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    cs = cases()
    n = len(cs) + 2
    made = [png(browser, doc(f"""
      <div style='padding:96px 64px;height:1150px;display:flex;flex-direction:column;justify-content:center'>
        <div class='pill red' style='align-self:flex-start'>Wallet security</div>
        <h1 style='font-size:76px;line-height:1.04;margin:30px 0;font-weight:800'>One signature can empty a wallet</h1>
        <p class='muted' style='font-size:32px;line-height:1.4'>Wallets show you hex and JSON. Drainers rely on that.
        I built a decoder that says, in plain English, what a request will actually do.</p>
        <p style='font-size:30px;margin-top:36px'>{len(cs)} real request formats →</p>
      </div>{FOOTER.format(page=f'1 / {n}')}""", W, H), out / "slide-1.png", W, H)]
    for i, c in enumerate(cs, 2):
        made.append(png(browser, _case_slide(c, i, n), out / f"slide-{i}.png", W, H))
    made.append(png(browser, doc(f"""
      <div style='padding:80px 64px'>
        <h1 style='font-size:58px;line-height:1.1;margin:0 0 40px;font-weight:800'>How it works</h1>
        <div style='font-size:30px;line-height:1.45'>
          <p><b>1. Decode in code.</b> <span class='muted'>Approvals, transfers, multicall, permits, Permit2, Seaport
          orders, eth_sign and sign-in messages. Exact amounts, addresses and expiries.</span></p>
          <p><b>2. Flag what drainers use.</b> <span class='muted'>Unlimited approvals, approval-for-all, zero-price
          listings, fake Permit2 contracts, blind signatures.</span></p>
          <p><b>3. Next: check intent.</b> <span class='muted'>A small model on my DGX Spark asks one question:
          does this match what you think you are doing?</span></p>
        </div>
        <p class='muted' style='font-size:24px;margin-top:40px'>Tested on local forks and testnets only. No real keys
        or funds.</p>
      </div>{FOOTER.format(page=f'{n} / {n}')}""", W, H), out / f"slide-{n}.png", W, H))
    pdf(browser, made, out / "carousel.pdf", W, H)
    return made


def _cli_output(args: list[str], names_file: Path) -> str:
    r = subprocess.run([sys.executable, "-m", "guard.wallet", "--names", str(names_file), *args],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip()


def video(browser, out: Path) -> Path:
    """A terminal typing three real decoder commands, with their real output."""
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        names_file = Path(td) / "names.json"
        names_file.write_text(json.dumps(NAMES))
        steps = []
        for c in [cases()[0], cases()[2], cases()[4]]:
            if "cli" in c:
                args = c["cli"]
                shown = f"python -m guard.wallet tx --to {args[2][:10]}… --data {args[4][:18]}…"
            else:
                f = Path(td) / "request.json"
                f.write_text(json.dumps(c["typed"]))
                args = ["typed", str(f)]
                shown = "python -m guard.wallet typed seaport-order.json"
            steps.append({"cmd": shown, "out": _cli_output(args, names_file), "title": c["title"]})
    w, h = 1280, 720
    html = doc(f"""
      <div style='position:absolute;inset:40px;background:#070b17;border-radius:18px;border:1px solid var(--line);
      overflow:hidden'>
        <div style='height:44px;background:#11182e;display:flex;align-items:center;gap:8px;padding:0 18px'>
          <span style='width:12px;height:12px;border-radius:6px;background:#f43f5e'></span>
          <span style='width:12px;height:12px;border-radius:6px;background:#f59e0b'></span>
          <span style='width:12px;height:12px;border-radius:6px;background:#22c55e'></span>
          <span class='muted' style='margin-left:16px;font-size:16px'>guard.wallet: what will this signature do?</span></div>
        <pre id='t' class='mono' style='margin:0;padding:22px 26px;font-size:19px;line-height:1.5;color:#dbe4ff;
        white-space:pre-wrap;height:590px;overflow:hidden'></pre></div>
      <script>
      const steps = {json.dumps(steps)};
      const t = document.getElementById('t');
      const sleep = ms => new Promise(r => setTimeout(r, ms));
      function color(line) {{
        const e = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;');
        if (line.startsWith('Risk: HIGH')) return '<span style="color:#f43f5e;font-weight:600">' + e(line) + '</span>';
        if (line.startsWith('Risk: MEDIUM')) return '<span style="color:#f59e0b;font-weight:600">' + e(line) + '</span>';
        if (line.startsWith('Risk: INFO')) return '<span style="color:#22c55e;font-weight:600">' + e(line) + '</span>';
        if (line.startsWith('[HIGH]')) return '<span style="color:#fb7185">' + e(line) + '</span>';
        if (line.startsWith('[MEDIUM]')) return '<span style="color:#fbbf24">' + e(line) + '</span>';
        return e(line);
      }}
      window.start = async () => {{
        for (const s of steps) {{
          t.innerHTML += '<span style="color:#7c8bb8"># ' + s.title + '</span>\\n<span style="color:#22c55e">$ </span>';
          for (const ch of s.cmd) {{ t.innerHTML += ch.replace('<','&lt;'); await sleep(28); }}
          t.innerHTML += '\\n'; await sleep(450);
          for (const line of s.out.split('\\n')) {{ t.innerHTML += color(line) + '\\n'; await sleep(260); }}
          t.innerHTML += '\\n'; await sleep(2600);
          t.scrollTop = t.scrollHeight;
        }}
      }};
      </script>""", w, h)
    seconds = sum(len(s["cmd"]) * 0.03 + 0.45 + 0.26 * len(s["out"].splitlines()) + 2.6 for s in steps) + 1.5
    return record(browser, html, out / "terminal", w, h, seconds)
