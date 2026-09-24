#!/usr/bin/env python3
"""Tiny shared Unix-socket JSON-line RPC server loop.

Protocol (must match `models/blanket/backend.py`'s `_RpcClient` in the
sibling `lose-the-faces-keep-the-lesson` repo exactly): one JSON object per
line, in each direction, over a single persistent connection (the client
connects once and reuses the socket for every call, it does not reconnect
per request).

    request:  {"op": "<name>", <kwarg>: <value>, ...}\n
    response: {"result": <value-or-null>}\n   on success (null is a
                                                legitimate "nothing found",
                                                e.g. no face detected)
              {"error": "<message>"}\n         on failure

Deliberately tiny and dependency-free (stdlib only) so both
`identity_server.py` and `swap_server.py` can import it regardless of which
venv they run in.
"""

from __future__ import annotations

import json
import os
import socket
import traceback
from typing import Callable


def serve(socket_path: str, handlers: dict[str, Callable[..., object]]) -> None:
    """Blocking accept loop. Handles one client connection at a time,
    multiple pipelined requests per connection (this project's own client
    keeps one connection open for the whole run, not one per call).
    """
    if os.path.exists(socket_path):
        os.remove(socket_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    print(f"[rpc_server] listening on {socket_path}", flush=True)

    while True:
        conn, _ = server.accept()
        print("[rpc_server] client connected", flush=True)
        buf = b""
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break  # client disconnected
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    _handle_line(conn, line, handlers)
        except (ConnectionResetError, BrokenPipeError) as e:
            print(f"[rpc_server] connection dropped: {e}", flush=True)
        finally:
            conn.close()
            print("[rpc_server] client disconnected", flush=True)


def _handle_line(conn: socket.socket, line: bytes, handlers: dict[str, Callable[..., object]]) -> None:
    try:
        request = json.loads(line.decode("utf-8"))
        op = request.pop("op", None)
        handler = handlers.get(op)
        if handler is None:
            reply = {"error": f"unknown op {op!r} (known: {sorted(handlers)})"}
        else:
            try:
                result = handler(**request)
                reply = {"result": result}
            except Exception as e:  # noqa: BLE001 - deliberately broad: any
                # handler failure must reach the client as a clear error,
                # never crash the server or silently return None.
                traceback.print_exc()
                reply = {"error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # malformed request line itself
        traceback.print_exc()
        reply = {"error": f"malformed request: {type(e).__name__}: {e}"}

    conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
