"""Just enough Solidity ABI decoding for the calls wallets are asked to sign."""

from __future__ import annotations

from .model import checksum

WORD = 32


class AbiError(ValueError):
    pass


def _word(data: bytes, offset: int) -> bytes:
    if offset < 0 or offset + WORD > len(data):
        raise AbiError(f"calldata too short: need bytes {offset}..{offset + WORD}, have {len(data)}")
    return data[offset:offset + WORD]


def _uint(data: bytes, offset: int) -> int:
    return int.from_bytes(_word(data, offset), "big")


def _decode_one(typ: str, data: bytes, head: int, base: int):
    """Decode one value of `typ` whose head word sits at `head`; dynamic offsets are relative to `base`."""
    if typ.endswith("[]"):
        inner = typ[:-2]
        start = base + _uint(data, head)
        n = _uint(data, start)
        if n > 10_000:
            raise AbiError("array too long")
        return [_decode_one(inner, data, start + WORD + i * WORD, start + WORD) for i in range(n)]
    if typ == "address":
        w = _word(data, head)
        if any(w[:12]):
            raise AbiError("address has non-zero high bytes")
        return checksum(w[12:].hex())
    if typ.startswith("uint"):
        return _uint(data, head)
    if typ == "bool":
        v = _uint(data, head)
        if v not in (0, 1):
            raise AbiError("bool is not 0 or 1")
        return bool(v)
    if typ == "bytes32":
        return "0x" + _word(data, head).hex()
    if typ == "bytes":
        start = base + _uint(data, head)
        n = _uint(data, start)
        body = data[start + WORD:start + WORD + n]
        if len(body) != n:
            raise AbiError("bytes runs past the end of calldata")
        return body
    raise AbiError(f"unsupported type {typ}")


def decode(types: list[str], data: bytes) -> list:
    """Decode an ABI-encoded tuple of `types` (calldata without the 4-byte selector)."""
    return [_decode_one(t, data, i * WORD, 0) for i, t in enumerate(types)]


def encode(types: list[str], values: list) -> bytes:
    """Encode values (test helper and fixtures). Supports the same types as `decode`."""
    heads, tails = [], []
    head_size = WORD * len(types)

    def enc_static(t, v) -> bytes:
        if t == "address":
            return bytes.fromhex(v.lower().removeprefix("0x")).rjust(WORD, b"\0")
        if t == "bool":
            return int(bool(v)).to_bytes(WORD, "big")
        if t == "bytes32":
            return bytes.fromhex(v.removeprefix("0x")).ljust(WORD, b"\0")
        return int(v).to_bytes(WORD, "big")

    def enc_dynamic(t, v) -> bytes:
        if t == "bytes":
            pad = (-len(v)) % WORD
            return len(v).to_bytes(WORD, "big") + v + b"\0" * pad
        inner = t[:-2]
        if inner in ("bytes",) or inner.endswith("[]"):
            parts = [enc_dynamic(inner, x) for x in v]
            offs, pos = [], WORD * len(v)
            for p in parts:
                offs.append(pos.to_bytes(WORD, "big"))
                pos += len(p)
            return len(v).to_bytes(WORD, "big") + b"".join(offs) + b"".join(parts)
        return len(v).to_bytes(WORD, "big") + b"".join(enc_static(inner, x) for x in v)

    for t, v in zip(types, values, strict=True):
        if t == "bytes" or t.endswith("[]"):
            heads.append((head_size + sum(len(x) for x in tails)).to_bytes(WORD, "big"))
            tails.append(enc_dynamic(t, v))
        else:
            heads.append(enc_static(t, v))
    return b"".join(heads) + b"".join(tails)
