"""Stand-in for the model endpoint the Chrome extension calls (`extension/`).

It answers POST /v1/page-check with fixed, rule-based verdicts so the extension can be built and
tested end to end before the guard model exists. The request and response shapes here are the
contract the real model server must keep.
"""

from .server import fake_verdicts, make_server

__all__ = ["fake_verdicts", "make_server"]
