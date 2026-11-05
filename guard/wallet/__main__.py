"""python -m guard.wallet {tx|typed|eth-sign|personal-sign} … [--json] [--names names.json]

  python -m guard.wallet tx --to 0xToken --data 0x095ea7b3… [--value 0] [--from 0xMe]
  python -m guard.wallet typed request.json          # eth_signTypedData_v4 payload
  python -m guard.wallet eth-sign 0x<32-byte hash>
  python -m guard.wallet personal-sign "<message or 0x hex>" [--origin app.example.org]

--names takes {"tokens": {addr: {"symbol": "USDC", "decimals": 6}}, "known": {addr: "My savings"}}.
Exit code: 0 low risk, 1 medium, 2 high.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import Names, explain_eth_sign, explain_personal_sign, explain_transaction, explain_typed_data

EXIT = {"info": 0, "medium": 1, "high": 2}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m guard.wallet", description="Explain a wallet request")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--names", type=Path, help="JSON file with token and address-book labels")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tx")
    t.add_argument("--to")
    t.add_argument("--data", default="0x")
    t.add_argument("--value", default="0")
    t.add_argument("--from", dest="sender")
    td = sub.add_parser("typed")
    td.add_argument("file", type=Path)
    es = sub.add_parser("eth-sign")
    es.add_argument("data")
    ps = sub.add_parser("personal-sign")
    ps.add_argument("message")
    ps.add_argument("--origin")
    a = ap.parse_args(argv)

    cfg = json.loads(a.names.read_text()) if a.names else {}
    names = Names(cfg.get("tokens"), cfg.get("known"))
    if a.cmd == "tx":
        out = explain_transaction({"to": a.to, "data": a.data, "value": a.value, "from": a.sender}, names)
    elif a.cmd == "typed":
        out = explain_typed_data(json.loads(a.file.read_text()), names)
    elif a.cmd == "eth-sign":
        out = explain_eth_sign(a.data)
    else:
        out = explain_personal_sign(a.message, a.origin)
    if a.json:
        print(json.dumps(out.to_dict(), indent=2))
    else:
        print(f"Risk: {out.risk.upper()}")
        print(out.plain())
    return EXIT[out.risk]


if __name__ == "__main__":
    sys.exit(main())
