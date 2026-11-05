"""Explain a wallet transaction request (to, data, value) as plain-English effects and risk flags."""

from __future__ import annotations

from .abi import AbiError, decode
from .keccak import selector
from .model import UNLIMITED, Decoded, Effect, Flag, Names, checksum, unknown_party_flag

# signature -> (kind, argument names). Selectors are computed from the signatures, not hard-coded.
_SIGNATURES = {
    "approve(address,uint256)": ("approve", ["spender", "amount"]),
    "increaseAllowance(address,uint256)": ("increase_allowance", ["spender", "amount"]),
    "decreaseAllowance(address,uint256)": ("decrease_allowance", ["spender", "amount"]),
    "setApprovalForAll(address,bool)": ("approve_all", ["operator", "approved"]),
    "transfer(address,uint256)": ("transfer", ["to", "amount"]),
    "transferFrom(address,address,uint256)": ("transfer_from", ["from", "to", "amount"]),
    "safeTransferFrom(address,address,uint256)": ("nft_transfer", ["from", "to", "id"]),
    "safeTransferFrom(address,address,uint256,bytes)": ("nft_transfer", ["from", "to", "id", "data"]),
    "safeTransferFrom(address,address,uint256,uint256,bytes)": (
        "erc1155_transfer", ["from", "to", "id", "amount", "data"]),
    "safeBatchTransferFrom(address,address,uint256[],uint256[],bytes)": (
        "erc1155_batch_transfer", ["from", "to", "ids", "amounts", "data"]),
    "permit(address,address,uint256,uint256,uint8,bytes32,bytes32)": (
        "permit_call", ["owner", "spender", "value", "deadline", "v", "r", "s"]),
    "multicall(bytes[])": ("multicall", ["calls"]),
    "multicall(uint256,bytes[])": ("multicall", ["deadline", "calls"]),
}
_TYPES = {sig: sig[sig.index("(") + 1:-1].split(",") for sig in _SIGNATURES}
SELECTORS = {selector(sig): sig for sig in _SIGNATURES}


def _hex(b: str) -> bytes:
    b = (b or "").removeprefix("0x")
    if len(b) % 2:
        raise ValueError("odd-length hex data")
    return bytes.fromhex(b)


def explain_transaction(tx: dict, names: Names | None = None, depth: int = 0) -> Decoded:
    """tx: {"to": addr, "data": "0x…", "value": int|hex str, "from": addr (optional)}."""
    names = names or Names()
    to = checksum(tx["to"]) if tx.get("to") else None
    sender = checksum(tx["from"]) if tx.get("from") else None
    value = tx.get("value") or 0
    value = int(value, 16) if isinstance(value, str) and value.startswith("0x") else int(value)
    data = _hex(tx.get("data", ""))
    effects: list[Effect] = []

    if to is None:
        effects.append(Effect("contract_deploy", "Deploys a new contract.",
                              flags=[Flag("contract_deploy", "medium", "This deploys new code from your account.")]))
        return Decoded("transaction", effects)

    if value:
        eth = value / 10**18
        flags = unknown_party_flag(names, to, "recipient")
        if data:
            flags.append(Flag("value_with_call", "medium", f"Sends {eth:,.6g} ETH along with a contract call; "
                                                           "the contract decides what happens to it."))
        effects.append(Effect("native_transfer", f"Sends {eth:,.6g} ETH to {names.who(to)}.",
                              counterparty=to, amount=value, flags=flags))
    if not data:
        if not value:
            effects.append(Effect("empty", f"Sends nothing to {names.who(to)} (an empty transaction)."))
        return Decoded("transaction", effects)

    sel = "0x" + data[:4].hex()
    sig = SELECTORS.get(sel)
    if sig is None:
        effects.append(Effect(
            "unknown_call", f"Calls an unrecognised function {sel} on {names.who(to)}.", counterparty=to,
            flags=[Flag("unknown_function", "medium",
                        "The wallet can't explain this call. Only continue if you trust this contract and site.")]))
        return Decoded("transaction", effects)

    kind, arg_names = _SIGNATURES[sig]
    try:
        args = dict(zip(arg_names, decode(_TYPES[sig], data[4:]), strict=True))
    except AbiError as exc:
        effects.append(Effect("malformed_call", f"Calls {sig} on {names.who(to)}, but the data is malformed.",
                              flags=[Flag("malformed", "high", f"The call data doesn't decode ({exc}).")]))
        return Decoded("transaction", effects)

    effects.extend(_explain_call(kind, args, to, sender, names, depth))
    return Decoded("transaction", effects)


def _explain_call(kind: str, a: dict, token: str, sender: str | None, names: Names, depth: int) -> list[Effect]:
    tok = names.token(token)
    if kind in ("approve", "increase_allowance"):
        spender, amount = a["spender"], a["amount"]
        unlimited = amount >= UNLIMITED
        verb = "Lets" if kind == "approve" else "Raises the allowance so"
        summary = (f"{verb} {names.who(spender)} spend {names.amount_of(token, amount)} from your wallet, "
                   "until you revoke it." if amount else f"Revokes {names.who(spender)}'s allowance on {tok}.")
        flags = []
        if unlimited:
            flags.append(Flag("unlimited_approval", "high",
                              f"{names.who(spender)} could take all of this token you hold now or later."))
        elif amount:
            flags.append(Flag("token_approval", "medium",
                              "An approval lets the spender move tokens later without asking you again. "
                              f"(On an NFT contract the same call approves NFT #{amount} instead.)"))
        if amount:
            flags += unknown_party_flag(names, spender, "spender")
        return [Effect(kind, summary, asset=token, counterparty=spender, amount=amount, unlimited=unlimited,
                       flags=flags)]
    if kind == "decrease_allowance":
        return [Effect(kind, f"Lowers {names.who(a['spender'])}'s allowance on {tok} by "
                             f"{names.amount_of(token, a['amount'])}.", asset=token, counterparty=a["spender"],
                       amount=a["amount"])]
    if kind == "approve_all":
        op = a["operator"]
        if not a["approved"]:
            return [Effect(kind, f"Revokes {names.who(op)}'s access to all your items in collection {short_c(token)}.",
                           asset=token, counterparty=op)]
        return [Effect(kind, f"Lets {names.who(op)} take ANY item you own in collection {short_c(token)}, "
                             "now or later, until you revoke it.", asset=token, counterparty=op, unlimited=True,
                       flags=[Flag("approval_for_all", "high",
                                   "Approval for all is how most NFT drainers empty a wallet in one go."),
                              *unknown_party_flag(names, op, "operator")])]
    if kind == "transfer":
        return [Effect(kind, f"Sends {names.amount_of(token, a['amount'])} to {names.who(a['to'])}.",
                       asset=token, counterparty=a["to"], amount=a["amount"],
                       flags=unknown_party_flag(names, a["to"], "recipient"))]
    if kind == "transfer_from":
        flags = unknown_party_flag(names, a["to"], "recipient")
        if sender and a["from"].lower() != sender.lower():
            flags.append(Flag("moves_others_funds", "medium",
                              f"Moves tokens out of {names.who(a['from'])}, which is not the sending account."))
        return [Effect(kind, f"Moves {names.amount_of(token, a['amount'])} from {names.who(a['from'])} "
                             f"to {names.who(a['to'])}.", asset=token, counterparty=a["to"], amount=a["amount"],
                       flags=flags)]
    if kind == "nft_transfer":
        return [Effect(kind, f"Sends NFT #{a['id']} of collection {short_c(token)} to {names.who(a['to'])}.",
                       asset=token, counterparty=a["to"], amount=1,
                       flags=unknown_party_flag(names, a["to"], "recipient"))]
    if kind == "erc1155_transfer":
        return [Effect(kind, f"Sends {a['amount']} of item #{a['id']} in collection {short_c(token)} "
                             f"to {names.who(a['to'])}.", asset=token, counterparty=a["to"], amount=a["amount"],
                       flags=unknown_party_flag(names, a["to"], "recipient"))]
    if kind == "erc1155_batch_transfer":
        n = len(a["ids"])
        return [Effect(kind, f"Sends {n} different items from collection {short_c(token)} to {names.who(a['to'])}.",
                       asset=token, counterparty=a["to"],
                       flags=[Flag("batch_transfer", "medium", f"Moves {n} item types in one transaction."),
                              *unknown_party_flag(names, a["to"], "recipient")])]
    if kind == "permit_call":
        unlimited = a["value"] >= UNLIMITED
        return [Effect(kind, f"Submits a signed permit letting {names.who(a['spender'])} spend "
                             f"{names.amount_of(token, a['value'])} owned by {names.who(a['owner'])}.",
                       asset=token, counterparty=a["spender"], amount=a["value"], unlimited=unlimited,
                       expires=a["deadline"],
                       flags=[Flag("unlimited_approval", "high", "The permit grants an unlimited allowance.")]
                       if unlimited else [])]
    if kind == "multicall":
        if depth >= 3:
            return [Effect("multicall", "Runs nested calls too deep to explain.",
                           flags=[Flag("deep_multicall", "medium", "Calls nested more than three levels deep.")])]
        out = [Effect("multicall", f"Runs {len(a['calls'])} calls on {short_c(token)} in one transaction:")]
        for raw in a["calls"]:
            inner = explain_transaction({"to": token, "data": "0x" + raw.hex(), "from": sender}, names, depth + 1)
            for e in inner.effects:
                e.summary = "  - " + e.summary
            out.extend(inner.effects)
        return out
    raise AssertionError(kind)


def short_c(addr: str) -> str:
    return f"{addr[:6]}…{addr[-4:]}"
