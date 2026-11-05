"""Result types shared by the transaction and signature decoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .keccak import keccak256

SEVERITY_ORDER = {"info": 0, "medium": 1, "high": 2}
UNLIMITED = 2**160 - 1  # Permit2 caps amounts at uint160; anything this large is effectively unlimited
MAX_UINT256 = 2**256 - 1


@dataclass
class Flag:
    code: str
    severity: str  # info | medium | high
    message: str


@dataclass
class Effect:
    kind: str
    summary: str
    asset: str | None = None
    counterparty: str | None = None
    amount: int | None = None
    unlimited: bool = False
    expires: int | None = None
    flags: list[Flag] = field(default_factory=list)


@dataclass
class Decoded:
    request: str  # transaction | typed_data | eth_sign | personal_sign
    effects: list[Effect]

    @property
    def risk(self) -> str:
        levels = [f.severity for e in self.effects for f in e.flags] or ["info"]
        return max(levels, key=SEVERITY_ORDER.__getitem__)

    @property
    def flags(self) -> list[Flag]:
        return [f for e in self.effects for f in e.flags]

    def plain(self) -> str:
        lines = [e.summary for e in self.effects]
        warnings = [f"[{f.severity.upper()}] {f.message}" for f in self.flags if f.severity != "info"]
        return "\n".join(lines + warnings)

    def to_dict(self) -> dict:
        d = {"request": self.request, "risk": self.risk, "plain": self.plain(), "effects": []}
        for e in self.effects:
            ed = asdict(e)
            for k in ("amount", "expires"):  # JSON numbers lose precision past 2**53
                if ed[k] is not None:
                    ed[k] = str(ed[k])
            d["effects"].append(ed)
        return d


def checksum(addr: str) -> str:
    """EIP-55 mixed-case checksum address."""
    a = addr.lower().removeprefix("0x")
    if len(a) != 40 or any(c not in "0123456789abcdef" for c in a):
        raise ValueError(f"not an address: {addr!r}")
    h = keccak256(a.encode()).hex()
    return "0x" + "".join(c.upper() if c.isalpha() and int(h[i], 16) >= 8 else c for i, c in enumerate(a))


def short(addr: str | None) -> str:
    if not addr:
        return "?"
    return f"{addr[:6]}…{addr[-4:]}"


class Names:
    """Labels for addresses the user already knows: tokens (symbol, decimals) and contacts/contracts."""

    def __init__(self, tokens: dict | None = None, known: dict | None = None):
        self.tokens = {k.lower(): v for k, v in (tokens or {}).items()}
        self.known = {k.lower(): v for k, v in (known or {}).items()}

    def token(self, addr: str | None) -> str:
        if not addr:
            return "an unknown token"
        t = self.tokens.get(addr.lower())
        return f"{t['symbol']} ({short(addr)})" if t else f"token {short(addr)}"

    def who(self, addr: str | None) -> str:
        if not addr:
            return "an unknown address"
        name = self.known.get(addr.lower())
        return f"{name} ({short(addr)})" if name else short(addr)

    def is_known(self, addr: str | None) -> bool:
        return bool(addr) and addr.lower() in self.known

    def amount(self, token: str | None, raw: int) -> str:
        if raw >= UNLIMITED:
            return "an UNLIMITED amount"
        t = self.tokens.get((token or "").lower())
        if t and "decimals" in t:
            value = raw / 10 ** int(t["decimals"])
            return f"{value:,.6g} {t['symbol']}"
        return f"{raw:,} base units"


def unknown_party_flag(names: Names, addr: str | None, role: str) -> list[Flag]:
    if names.known and not names.is_known(addr):
        return [Flag("unknown_counterparty", "medium", f"The {role} {short(addr)} is not in your address book.")]
    return []
