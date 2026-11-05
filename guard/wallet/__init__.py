"""Explain wallet requests in plain English before they are signed.

Code does everything exact here: decoding, amounts, addresses, expiries. A guard model later
answers the judgment call code can't: does this effect match what the user thinks they are doing?
"""

from .calldata import explain_transaction
from .model import Decoded, Effect, Flag, Names, checksum
from .signatures import PERMIT2, explain_eth_sign, explain_personal_sign, explain_typed_data

__all__ = [
    "PERMIT2", "Decoded", "Effect", "Flag", "Names", "checksum", "explain_eth_sign", "explain_personal_sign",
    "explain_transaction", "explain_typed_data",
]
