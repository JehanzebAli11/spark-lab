"""Explain off-chain signature requests: EIP-712 typed data, eth_sign and personal_sign.

Off-chain signatures cost no gas and show up nowhere until someone submits them, which is why
drainers prefer them: one signature on a Permit, Permit2 or Seaport order can hand over tokens.
"""

from __future__ import annotations

import re
import time

from .model import UNLIMITED, Decoded, Effect, Flag, Names, checksum, short, unknown_party_flag

PERMIT2 = "0x000000000022D473030F116dDEE9F6B43aC78BA3"  # Uniswap Permit2, same address on every chain
YEAR = 365 * 24 * 3600

# Seaport item types
_NATIVE, _ERC20, _ERC721, _ERC1155, _ERC721_CRITERIA, _ERC1155_CRITERIA = range(6)


def _int(v) -> int:
    if isinstance(v, str):
        return int(v, 16) if v.startswith("0x") else int(v)
    return int(v)


def _addr(v) -> str:
    return checksum(v)


def _expiry_flags(ts: int | None, now: int, what: str) -> list[Flag]:
    if ts is None:
        return []
    if ts == 0 or ts >= 2**47:
        return [Flag("no_expiry", "medium", f"The {what} never expires.")]
    if ts - now > YEAR:
        return [Flag("far_expiry", "medium", f"The {what} lasts more than a year.")]
    return []


def explain_typed_data(td: dict, names: Names | None = None, now: int | None = None) -> Decoded:
    """td: the eth_signTypedData_v4 payload {types, domain, primaryType, message}."""
    names = names or Names()
    now = int(time.time()) if now is None else now
    domain, primary, msg = td.get("domain", {}), td.get("primaryType", ""), td.get("message", {})
    dname = str(domain.get("name", ""))
    contract = domain.get("verifyingContract")
    contract = _addr(contract) if contract else None
    base_flags = [Flag("offchain_signature", "info",
                       "This is a signature, not a transaction: it costs no gas and won't show in your history, "
                       "but whoever holds it can submit it later.")]

    if dname == "Permit2" and contract and contract.lower() != PERMIT2.lower():
        base_flags.append(Flag("fake_permit2", "high",
                               f"Claims to be Permit2 but the verifying contract is {short(contract)}, "
                               f"not the real Permit2 ({short(PERMIT2)})."))

    if primary == "Permit" and "spender" in msg:
        effects = _eip2612(msg, contract, names, now)
    elif primary in ("PermitSingle", "PermitBatch"):
        effects = _permit2_allowance(primary, msg, names, now)
    elif primary in ("PermitTransferFrom", "PermitBatchTransferFrom", "PermitWitnessTransferFrom",
                     "PermitBatchWitnessTransferFrom"):
        effects = _permit2_transfer(msg, names, now)
    elif primary in ("OrderComponents", "Order") and "offer" in msg:
        effects = _seaport(msg, dname, names)
    else:
        effects = [Effect("unknown_typed_data",
                          f"Signs a '{primary or 'unnamed'}' message for {dname or 'an unnamed app'}.",
                          counterparty=contract,
                          flags=[Flag("unknown_typed_data", "medium",
                                      "The wallet can't explain this message type. Read every field before signing.")])]
    effects[0].flags = base_flags + effects[0].flags
    return Decoded("typed_data", effects)


def _eip2612(m: dict, token: str | None, names: Names, now: int) -> list[Effect]:
    spender = _addr(m["spender"])
    if "allowed" in m:  # DAI-style permit: all or nothing
        allowed = bool(m["allowed"])
        expiry = _int(m.get("expiry", 0))
        if not allowed:
            return [Effect("permit", f"Signs away {names.who(spender)}'s permission on {names.token(token)}.",
                           asset=token, counterparty=spender)]
        return [Effect("permit", f"Lets {names.who(spender)} spend an UNLIMITED amount of your {names.token(token)}.",
                       asset=token, counterparty=spender, unlimited=True, expires=expiry,
                       flags=[Flag("unlimited_approval", "high", "A DAI-style permit gives an unlimited allowance."),
                              *_expiry_flags(expiry, now, "permission"),
                              *unknown_party_flag(names, spender, "spender")])]
    value, deadline = _int(m["value"]), _int(m.get("deadline", 0))
    unlimited = value >= UNLIMITED
    flags = [Flag("unlimited_approval", "high",
                  f"{names.who(spender)} could take all of this token you hold.")] if unlimited else [
        Flag("token_approval", "medium", "Signing grants an allowance the spender can use without asking again.")]
    flags += _expiry_flags(deadline, now, "signature") + unknown_party_flag(names, spender, "spender")
    return [Effect("permit", f"Lets {names.who(spender)} spend {names.amount(token, value)} of your "
                             f"{names.token(token)}.", asset=token, counterparty=spender, amount=value,
                   unlimited=unlimited, expires=deadline, flags=flags)]


def _permit2_allowance(primary: str, m: dict, names: Names, now: int) -> list[Effect]:
    spender = _addr(m["spender"])
    details = m["details"] if primary == "PermitSingle" else None
    items = [details] if details else list(m["details"])
    out = []
    for d in items:
        token, amount, expiration = _addr(d["token"]), _int(d["amount"]), _int(d.get("expiration", 0))
        unlimited = amount >= UNLIMITED
        flags = [Flag("unlimited_approval", "high", f"{names.who(spender)} could take all your {names.token(token)}.")
                 ] if unlimited else [Flag("token_approval", "medium", "Grants a Permit2 allowance.")]
        flags += _expiry_flags(expiration, now, "allowance") + unknown_party_flag(names, spender, "spender")
        out.append(Effect("permit2_allowance", f"Lets {names.who(spender)} spend {names.amount(token, amount)} "
                                               f"of your {names.token(token)} through Permit2.",
                          asset=token, counterparty=spender, amount=amount, unlimited=unlimited,
                          expires=expiration, flags=flags))
    if len(out) > 1:
        out[0].flags.append(Flag("batch_approval", "high", f"Approves {len(out)} tokens with one signature."))
    return out


def _permit2_transfer(m: dict, names: Names, now: int) -> list[Effect]:
    spender = _addr(m["spender"])
    permitted = m["permitted"]
    items = permitted if isinstance(permitted, list) else [permitted]
    deadline = _int(m.get("deadline", 0))
    out = []
    for p in items:
        token, amount = _addr(p["token"]), _int(p["amount"])
        out.append(Effect("permit2_transfer",
                          f"Lets {names.who(spender)} take {names.amount(token, amount)} of your "
                          f"{names.token(token)} directly, once.", asset=token, counterparty=spender, amount=amount,
                          unlimited=amount >= UNLIMITED, expires=deadline,
                          flags=[Flag("signature_transfer", "high",
                                      "This signature lets the spender move the tokens itself, without a "
                                      "transaction from you."),
                                 *unknown_party_flag(names, spender, "spender")]))
    out[0].flags += _expiry_flags(deadline, now, "signature")
    return out


def _seaport(m: dict, dname: str, names: Names) -> list[Effect]:
    offerer = _addr(m["offerer"])
    offer = m.get("offer", [])
    consideration = m.get("consideration", [])
    gives = []
    for it in offer:
        t, token = _int(it["itemType"]), _addr(it["token"]) if _int(it["itemType"]) else None
        if t in (_ERC721_CRITERIA, _ERC1155_CRITERIA):
            gives.append(f"ANY item from collection {short(token)}")
        elif t in (_ERC721, _ERC1155):
            gives.append(f"item #{_int(it['identifierOrCriteria'])} of {short(token)}")
        elif t == _ERC20:
            gives.append(f"{names.amount(token, _int(it['startAmount']))} of {names.token(token)}")
        else:
            gives.append(f"{_int(it['startAmount']) / 1e18:,.6g} ETH")
    to_you = sum(_int(c["startAmount"]) for c in consideration
                 if _int(c["itemType"]) in (_NATIVE, _ERC20) and _addr(c["recipient"]).lower() == offerer.lower())
    others = {_addr(c["recipient"]) for c in consideration if _addr(c["recipient"]).lower() != offerer.lower()}
    flags = []
    if to_you == 0 and offer:
        flags.append(Flag("zero_price_listing", "high",
                          "You receive nothing in return. This is the classic Seaport drainer signature."))
    if any(_int(it["itemType"]) in (_ERC721_CRITERIA, _ERC1155_CRITERIA) for it in offer):
        flags.append(Flag("collection_wide", "high", "The order can be filled with any item from the collection."))
    if len(offer) > 3:
        flags.append(Flag("bulk_listing", "medium", f"Lists {len(offer)} items in one signature."))
    paid = f"{to_you / 1e18:,.6g} ETH (or token units)" if to_you else "nothing"
    summary = (f"Signs a {dname or 'marketplace'} order: you give {', '.join(gives) or 'nothing'}; "
               f"you receive {paid}. Other recipients: {', '.join(short(o) for o in sorted(others)) or 'none'}.")
    return [Effect("seaport_order", summary, counterparty=None, flags=flags)]


_HEX_RE = re.compile(r"^(0x)?[0-9a-fA-F]+$")


def explain_eth_sign(data: str) -> Decoded:
    """eth_sign over raw data: the user cannot see what they are approving."""
    blind = bool(_HEX_RE.match(data or "")) and len(data.removeprefix("0x")) == 64
    msg = "Signs a raw 32-byte hash you can't read." if blind else "Signs raw data with eth_sign."
    return Decoded("eth_sign", [Effect("blind_signature", msg, flags=[Flag(
        "blind_signature", "high",
        "eth_sign can sign a transaction or permit in disguise. Legitimate apps almost never need it.")])])


_SIWE = re.compile(r"^(?P<domain>\S+) wants you to sign in with your Ethereum account", re.MULTILINE)


def explain_personal_sign(message: str, origin: str | None = None) -> Decoded:
    """personal_sign: hex-encoded or plain text. `origin` is the site asking (e.g. 'app.example.org')."""
    text = message
    if _HEX_RE.match(message or "") and message.startswith("0x"):
        raw = bytes.fromhex(message[2:])
        if len(raw) == 32:
            return Decoded("personal_sign", [Effect("blind_signature", "Signs a 32-byte value you can't read.",
                                                    flags=[Flag("blind_signature", "high",
                                                                "This looks like a hash, not a message. "
                                                                "Don't sign data you can't read.")])])
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            flag = Flag("unreadable_message", "medium", "The message isn't text; check why the site needs it.")
            return Decoded("personal_sign", [Effect("binary_message", "Signs binary data that isn't readable text.",
                                                    flags=[flag])])
    m = _SIWE.search(text)
    if m:
        domain = m.group("domain")
        flags = []
        if origin and domain.lower() != origin.lower():
            flags.append(Flag("sign_in_domain_mismatch", "high",
                              f"The sign-in message names {domain}, but the request comes from {origin}."))
        return Decoded("personal_sign", [Effect("sign_in", f"Signs in to {domain} with your address.", flags=flags)])
    preview = text if len(text) <= 120 else text[:117] + "..."
    return Decoded("personal_sign", [Effect("message", f'Signs the text message: "{preview}"')])
