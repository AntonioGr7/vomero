"""Length-prefixed JSON framing for the host <-> sandbox control channel.

A tiny, dependency-free wire format shared by the host (`SandboxEnvironment`)
and the in-sandbox `agent.py`. Each message is a 4-byte big-endian length
followed by that many UTF-8 bytes of JSON. Both ends speak strictly synchronous
request/response (one outstanding message at a time, in each direction), so
frames never interleave and we don't need ids or multiplexing.

The agent runs standalone inside the sandbox and imports this module by sitting
next to it on `sys.path` — so keep it stdlib-only and self-contained.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

# 4-byte unsigned big-endian payload length prefix.
_HEADER = struct.Struct(">I")


def send_msg(sock: socket.socket, obj: Any) -> None:
    """Frame and send one JSON message."""
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(_HEADER.pack(len(data)) + data)


def recv_msg(sock: socket.socket) -> Any | None:
    """Receive one framed message, or None on a clean EOF (peer closed).

    Returning None rather than raising lets both ends treat "the other side
    went away" as an ordinary, handleable event (container died, host closed)."""
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    body = _recv_exactly(sock, length)
    if body is None:  # truncated mid-message — treat as a disconnect
        return None
    return json.loads(body.decode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly `n` bytes; None if the peer closed before sending them."""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
