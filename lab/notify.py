"""Phone notifications through ntfy (https://ntfy.sh or a self-hosted server).

Set NTFY_TOPIC (required to send) and optionally NTFY_SERVER (default https://ntfy.sh)
and NTFY_TOKEN. With no topic set, notifications are printed and skipped, never fatal.
Pick a long random topic name: anyone who knows it can read the messages.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from collections.abc import Callable

Sender = Callable[[urllib.request.Request], object]


def _default_send(req: urllib.request.Request) -> object:
    return urllib.request.urlopen(req, timeout=10)


def notify(
    title: str,
    message: str,
    *,
    priority: str = "default",
    tags: str = "",
    send: Sender | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    """Send a notification. Returns True if a request was sent successfully."""
    env = dict(os.environ) if env is None else env
    topic = env.get("NTFY_TOPIC", "").strip()
    if not topic:
        print(f"[notify skipped: NTFY_TOPIC unset] {title}: {message}", file=sys.stderr)
        return False
    server = env.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    req = urllib.request.Request(f"{server}/{topic}", data=message.encode("utf-8"), method="POST")
    # ntfy headers must be latin-1; keep titles ASCII-safe.
    req.add_header("Title", title.encode("ascii", "replace").decode("ascii"))
    req.add_header("Priority", priority)
    if tags:
        req.add_header("Tags", tags)
    token = env.get("NTFY_TOKEN", "").strip()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        (send or _default_send)(req)
        return True
    except Exception as exc:  # noqa: BLE001 - notifications must never break a job
        print(f"[notify failed] {exc}", file=sys.stderr)
        return False
