import json

import pytest

from guard.wallet import (
    PERMIT2,
    Names,
    checksum,
    explain_eth_sign,
    explain_personal_sign,
    explain_transaction,
    explain_typed_data,
)
from guard.wallet.__main__ import main as cli
from guard.wallet.abi import AbiError, decode, encode
from guard.wallet.keccak import keccak256, selector

USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
ME = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
EVIL = "0x1111111111111111111111111111111111111111"
FRIEND = "0x2222222222222222222222222222222222222222"
NFT = "0x3333333333333333333333333333333333333333"
MAX = 2**256 - 1
NOW = 1_800_000_000


def call(sig: str, types: list[str], values: list) -> str:
    return selector(sig) + encode(types, values).hex()


def codes(decoded) -> set[str]:
    return {f.code for f in decoded.flags}


# --- primitives -------------------------------------------------------------------------------

def test_keccak_known_vectors():
    assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"


@pytest.mark.parametrize(("sig", "sel"), [
    ("approve(address,uint256)", "0x095ea7b3"),
    ("transfer(address,uint256)", "0xa9059cbb"),
    ("transferFrom(address,address,uint256)", "0x23b872dd"),
    ("setApprovalForAll(address,bool)", "0xa22cb465"),
    ("permit(address,address,uint256,uint256,uint8,bytes32,bytes32)", "0xd505accf"),
])
def test_selectors_match_published_values(sig, sel):
    assert selector(sig) == sel


def test_eip55_checksum_vectors():
    for a in ["0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
              "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB", "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb"]:
        assert checksum(a.lower()) == a


def test_abi_roundtrip_with_dynamic_types():
    types = ["address", "uint256[]", "bytes", "bytes[]", "bool"]
    values = [EVIL, [1, 2, 3], b"hello", [b"a", b"bc" * 40], True]
    assert decode(types, encode(types, values)) == [checksum(EVIL), [1, 2, 3], b"hello", [b"a", b"bc" * 40], True]


def test_abi_rejects_truncated_and_dirty_data():
    with pytest.raises(AbiError):
        decode(["address", "uint256"], b"\0" * 40)
    with pytest.raises(AbiError):
        decode(["address"], b"\x01" * 32)


# --- transactions -----------------------------------------------------------------------------

def test_unlimited_approval_is_high_risk():
    d = explain_transaction({"to": USDC, "data": call("approve(address,uint256)", ["address", "uint256"], [EVIL, MAX])})
    assert d.risk == "high"
    assert "unlimited_approval" in codes(d)
    assert "UNLIMITED" in d.plain()
    assert d.effects[0].counterparty == checksum(EVIL)


def test_limited_approval_shows_token_amount_with_decimals():
    names = Names(tokens={USDC: {"symbol": "USDC", "decimals": 6}}, known={FRIEND: "Uniswap router"})
    d = explain_transaction({"to": USDC, "data": call("approve(address,uint256)", ["address", "uint256"],
                                                      [FRIEND, 25_000_000])}, names)
    assert d.risk == "medium"
    assert "25 USDC" in d.plain() and "Uniswap router" in d.plain()
    assert "unknown_counterparty" not in codes(d)


def test_unknown_spender_flagged_when_address_book_given():
    names = Names(known={FRIEND: "Friend"})
    d = explain_transaction({"to": USDC, "data": call("approve(address,uint256)", ["address", "uint256"],
                                                      [EVIL, 5])}, names)
    assert "unknown_counterparty" in codes(d)


def test_zero_approval_is_a_revoke():
    d = explain_transaction({"to": USDC, "data": call("approve(address,uint256)", ["address", "uint256"], [EVIL, 0])})
    assert d.risk == "info"
    assert "Revokes" in d.plain()


def test_set_approval_for_all():
    on = explain_transaction({"to": NFT, "data": call("setApprovalForAll(address,bool)", ["address", "bool"],
                                                      [EVIL, True])})
    off = explain_transaction({"to": NFT, "data": call("setApprovalForAll(address,bool)", ["address", "bool"],
                                                       [EVIL, False])})
    assert on.risk == "high" and "approval_for_all" in codes(on) and "ANY item" in on.plain()
    assert off.risk == "info" and "Revokes" in off.plain()


def test_transfer_from_someone_else():
    d = explain_transaction({"to": USDC, "from": ME, "data": call(
        "transferFrom(address,address,uint256)", ["address", "address", "uint256"], [FRIEND, EVIL, 10])})
    assert "moves_others_funds" in codes(d)


def test_multicall_is_unpacked():
    inner1 = bytes.fromhex(call("approve(address,uint256)", ["address", "uint256"], [EVIL, MAX])[2:])
    inner2 = bytes.fromhex(call("transfer(address,uint256)", ["address", "uint256"], [FRIEND, 7])[2:])
    d = explain_transaction({"to": USDC, "data": call("multicall(bytes[])", ["bytes[]"], [[inner1, inner2]])})
    kinds = [e.kind for e in d.effects]
    assert kinds == ["multicall", "approve", "transfer"]
    assert d.risk == "high"


def test_unknown_function_and_value():
    d = explain_transaction({"to": EVIL, "data": "0xdeadbeef" + "00" * 32, "value": "0xde0b6b3a7640000"})
    assert {"unknown_function", "value_with_call"} <= codes(d)
    assert "Sends 1 ETH" in d.plain()


def test_plain_eth_send_and_malformed_call():
    assert explain_transaction({"to": FRIEND, "value": 10**17}).risk == "info"
    bad = explain_transaction({"to": USDC, "data": selector("approve(address,uint256)") + "00" * 10})
    assert bad.risk == "high" and "malformed" in codes(bad)


# --- typed data -------------------------------------------------------------------------------

def permit2_single(amount, verifying=PERMIT2, expiration=NOW + 3600):
    return {
        "domain": {"name": "Permit2", "chainId": 1, "verifyingContract": verifying},
        "primaryType": "PermitSingle",
        "message": {"details": {"token": USDC, "amount": str(amount), "expiration": str(expiration), "nonce": "0"},
                    "spender": EVIL, "sigDeadline": str(NOW + 600)},
    }


def test_permit2_unlimited_allowance():
    d = explain_typed_data(permit2_single(2**160 - 1), now=NOW)
    assert d.risk == "high" and "unlimited_approval" in codes(d) and "offchain_signature" in codes(d)


def test_permit2_with_wrong_contract_is_fake():
    d = explain_typed_data(permit2_single(5, verifying=EVIL), now=NOW)
    assert "fake_permit2" in codes(d)


def test_permit2_long_expiry():
    d = explain_typed_data(permit2_single(5, expiration=NOW + 3 * 365 * 24 * 3600), now=NOW)
    assert "far_expiry" in codes(d)


def test_permit2_signature_transfer_batch():
    td = {"domain": {"name": "Permit2", "verifyingContract": PERMIT2}, "primaryType": "PermitBatchTransferFrom",
          "message": {"permitted": [{"token": USDC, "amount": "100"}, {"token": NFT, "amount": "5"}],
                      "spender": EVIL, "nonce": "1", "deadline": str(NOW + 60)}}
    d = explain_typed_data(td, now=NOW)
    assert [e.kind for e in d.effects] == ["permit2_transfer", "permit2_transfer"]
    assert "signature_transfer" in codes(d)


def test_eip2612_and_dai_permits():
    eip = {"domain": {"name": "USD Coin", "verifyingContract": USDC}, "primaryType": "Permit",
           "message": {"owner": ME, "spender": EVIL, "value": str(MAX), "nonce": "0", "deadline": str(NOW + 60)}}
    dai = {"domain": {"name": "Dai Stablecoin", "verifyingContract": NFT}, "primaryType": "Permit",
           "message": {"holder": ME, "spender": EVIL, "nonce": 0, "expiry": 0, "allowed": True}}
    assert "unlimited_approval" in codes(explain_typed_data(eip, now=NOW))
    d = explain_typed_data(dai, now=NOW)
    assert {"unlimited_approval", "no_expiry"} <= codes(d)


def seaport(consideration):
    return {"domain": {"name": "Seaport", "version": "1.6"}, "primaryType": "OrderComponents",
            "message": {"offerer": ME,
                        "offer": [{"itemType": 2, "token": NFT, "identifierOrCriteria": "42",
                                   "startAmount": "1", "endAmount": "1"}],
                        "consideration": consideration}}


def test_seaport_zero_price_listing_is_high():
    d = explain_typed_data(seaport([{"itemType": 0, "token": "0x" + "0" * 40, "identifierOrCriteria": "0",
                                     "startAmount": "1000", "endAmount": "1000", "recipient": EVIL}]), now=NOW)
    assert "zero_price_listing" in codes(d) and d.risk == "high"


def test_seaport_fair_listing_is_not_flagged():
    d = explain_typed_data(seaport([{"itemType": 0, "token": "0x" + "0" * 40, "identifierOrCriteria": "0",
                                     "startAmount": str(10**18), "endAmount": str(10**18), "recipient": ME}]), now=NOW)
    assert "zero_price_listing" not in codes(d)
    assert "you receive 1 ETH" in d.plain()


def test_unknown_typed_data():
    d = explain_typed_data({"domain": {"name": "Some Game"}, "primaryType": "Move", "message": {"x": 1}}, now=NOW)
    assert d.risk == "medium" and "unknown_typed_data" in codes(d)


# --- raw signatures ---------------------------------------------------------------------------

def test_eth_sign_hash_is_blind():
    assert explain_eth_sign("0x" + "ab" * 32).risk == "high"


def test_personal_sign_variants():
    hello = "0x" + b"Hello, sign to log in".hex()
    assert explain_personal_sign(hello).risk == "info"
    assert explain_personal_sign("0x" + "cd" * 32).risk == "high"
    siwe = "evil.example wants you to sign in with your Ethereum account:\n" + ME
    assert "sign_in_domain_mismatch" in codes(explain_personal_sign(siwe, origin="app.uniswap.org"))
    assert explain_personal_sign(siwe, origin="evil.example").risk == "info"


# --- CLI --------------------------------------------------------------------------------------

def test_cli_exit_codes_and_json(tmp_path, capsys):
    data = call("approve(address,uint256)", ["address", "uint256"], [EVIL, MAX])
    assert cli(["--json", "tx", "--to", USDC, "--data", data]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["risk"] == "high" and out["effects"][0]["amount"] == str(MAX)
    f = tmp_path / "td.json"
    f.write_text(json.dumps(permit2_single(5)))
    assert cli(["typed", str(f)]) in (1, 2)
    assert cli(["personal-sign", "hi there"]) == 0
