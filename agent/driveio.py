"""Crash-safe, sync-safe I/O over a Google Drive mount.

STDLIB ONLY - mirrored verbatim into ``colab-bus/agent/driveio.py``.

Two rules this module exists to enforce:

1. **Never append to one growing file on a FUSE mount.** Every flush re-uploads the whole
   file from Colab and re-downloads the whole file onto ``H:``. Logs are written as
   immutable numbered chunks instead.
2. **Never trust a file you merely found.** A chunk that is still syncing looks like a
   short file, not like an error. Every chunk therefore ends with a sentinel line that
   names its own sequence number, length and content hash; a reader that cannot verify
   the sentinel skips the chunk and retries on the next poll. Partial reads become
   impossible to misinterpret rather than merely unlikely.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

SENTINEL_PREFIX = "#EOC"
_SENTINEL_RE = re.compile(r"^#EOC (\d+) ([0-9a-f]{8}) (\d+)$")
CHUNK_GLOB = "[0-9]" * 6 + ".txt"


def payload_digest(payload: str) -> str:
    """First 8 hex chars of the sha1 of a chunk payload."""
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def write_chunk(log_dir: Path, seq: int, payload: str) -> Path:
    """Write one immutable, self-verifying log chunk.

    Written as ``NNNNNN.txt.part`` then renamed, so a reader never observes a
    half-written file under the final name.

    Args:
        log_dir: directory holding the chunks (created if absent).
        seq: 1-based chunk sequence number.
        payload: the log text; a trailing newline is added if missing.

    Returns:
        Path to the finished chunk.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    if payload and not payload.endswith("\n"):
        payload += "\n"
    n_lines = payload.count("\n")
    sentinel = f"{SENTINEL_PREFIX} {seq} {payload_digest(payload)} {n_lines}\n"
    final = log_dir / f"{seq:06d}.txt"
    tmp = log_dir / f"{seq:06d}.txt.part"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload)
        fh.write(sentinel)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return final


def read_chunk(path: Path) -> str | None:
    """Return a chunk's payload, or ``None`` if it is incomplete or corrupt.

    ``None`` is the normal answer for a chunk that is still syncing - the caller should
    simply try again later, not treat it as an error.

    Args:
        path: chunk file, named ``NNNNNN.txt``.

    Returns:
        The payload text without the sentinel line, or ``None``.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    if not text.endswith("\n"):
        return None
    lines = text.split("\n")
    # text ends with "\n" so the final element is "" and the sentinel is the one before it
    sentinel = lines[-2] if len(lines) >= 2 else ""
    m = _SENTINEL_RE.match(sentinel)
    if not m:
        return None
    seq, digest, n_lines = int(m.group(1)), m.group(2), int(m.group(3))
    if seq != int(path.stem):
        return None
    payload = "\n".join(lines[:-2])
    if payload:
        payload += "\n"
    if payload.count("\n") != n_lines or payload_digest(payload) != digest:
        return None
    return payload


def iter_chunks(log_dir: Path, since_seq: int = 0) -> Iterator[tuple[int, str]]:
    """Yield ``(seq, payload)`` for every *complete* chunk after ``since_seq``, in order.

    Stops at the first gap: if chunk 7 is unreadable, chunk 8 is not yielded either, so
    the caller never sees log lines out of order or with a hole in the middle.

    Args:
        log_dir: directory holding the chunks.
        since_seq: yield chunks with ``seq > since_seq``.
    """
    if not log_dir.is_dir():
        return
    seqs = sorted(int(p.stem) for p in log_dir.glob(CHUNK_GLOB))
    for seq in seqs:
        if seq <= since_seq:
            continue
        payload = read_chunk(log_dir / f"{seq:06d}.txt")
        if payload is None:
            return  # incomplete - stop here and let the caller retry
        yield seq, payload


def write_json_atomic(path: Path, obj: Any) -> Path:
    """Serialize ``obj`` to ``path`` via a temp file + rename.

    Args:
        path: destination ``.json`` file.
        obj: anything ``json.dumps`` accepts.

    Returns:
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> Any | None:
    """Read a JSON file, returning ``None`` if it is missing or not yet fully synced."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def newest_mtime(root: Path) -> float | None:
    """Newest mtime under ``root``, or ``None`` if nothing is there.

    This is the liveness signal: the agent is alive iff something under its run directory
    was touched recently. No separate heartbeat file to go stale on its own.
    """
    newest: float | None = None
    if not root.is_dir():
        return None
    for p in root.rglob("*"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if newest is None or m > newest:
            newest = m
    return newest
