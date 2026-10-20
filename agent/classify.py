"""Turn a failed job's log into a structured, actionable failure.

STDLIB ONLY - mirrored verbatim into ``colab-bus/agent/classify.py`` so both sides
classify identically. The Colab agent classifies at write time; the local driver can
re-classify from ``full.log`` for a second opinion. One implementation, no drift.

The classification is what the auto-fix policy dispatches on, so it must be *specific*:
"failed" is useless, ``OOM_CUDA`` tells you to halve the batch. When nothing matches,
say ``UNKNOWN`` rather than guessing - a wrong class sends the auto-fix loop down a path
that cannot possibly work and burns the retry budget doing it.
"""

from __future__ import annotations

import re

# Ordered most-specific first: the first pattern that matches wins, so a CUDA OOM is
# never demoted to a generic CODE_ERROR just because it also raised a RuntimeError.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OOM_CUDA", re.compile(
        r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED"
        r"|DefaultCPUAllocator: can't allocate memory", re.I)),
    ("DISK_FULL", re.compile(r"No space left on device|\[Errno 28\]|Disk quota exceeded", re.I)),
    ("DRIVE_IO", re.compile(
        r"Transport endpoint is not connected|Input/output error"
        r"|OSError:.*/content/drive|drive\.mount.*failed|MessageError:.*mount", re.I)),
    ("MODEL_NOT_FOUND", re.compile(
        r"RepositoryNotFoundError|is not a valid model identifier"
        r"|404 Client Error.*huggingface|GatedRepoError|EntryNotFoundError", re.I)),
    ("USAGE_LIMIT", re.compile(
        r"You cannot currently connect to a GPU|usage limit|compute units"
        r"|ResourceExhausted|exceeded your .* quota", re.I)),
    ("GPU_ABSENT", re.compile(
        r"NVIDIA-SMI has failed|no CUDA-capable device|cuda\.is_available\(\) *(?:is *)?False"
        r"|GPU_NOT_AVAILABLE|nvidia-smi: not found", re.I)),
    ("DEPENDENCY_CONFLICT", re.compile(
        r"cannot import name|No module named|has no attribute '__version__'"
        r"|requires .* but you have|incompatible|ImportError|ModuleNotFoundError"
        r"|pip's dependency resolver", re.I)),
    ("DATA_MISSING", re.compile(
        r"metadata\.csv not found|FileNotFoundError|No such file or directory"
        r"|BadZipFile|does not appear to have a file named", re.I)),
]

# Warnings are deliberately NOT matched. This scans backwards for the LAST exception
# line, so a warning printed after the real traceback would shadow the actual cause -
# observed 2026-09-16, when a cancelled job recorded an HF Hub rate-limit warning as
# its message instead of the reason it stopped.
_EXC_RE = re.compile(r"^(?:\w+\.)*(\w*(?:Error|Exception|Exit|Interrupt))\s*:\s*(.*)$")
_FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+)')

#: A traceback frame under one of these path fragments is code we wrote and may fix.
_OURS_MARKERS = ("/payload/", "\\payload\\", "/agent/", "\\agent\\", "colab-bus", "/hygdap/", "\\hygdap\\")


def _last_exception(lines: list[str]) -> tuple[str, str, int | None]:
    """Find the last ``ExcType: message`` line - the one that actually killed the job.

    Args:
        lines: log lines, oldest first.

    Returns:
        ``(exception_type, message, line_no)``; ``("", "", None)`` when there is no
        Python exception (a non-zero exit with no traceback, say).
    """
    for i in range(len(lines) - 1, -1, -1):
        m = _EXC_RE.match(lines[i].rstrip())
        if m:
            return m.group(1), m.group(2).strip(), i + 1
    return "", "", None


def _origin(lines: list[str]) -> tuple[str, str]:
    """Decide whether the deepest traceback frame is our code or a third party's.

    This single field gates the auto-fix loop: we may edit ``payload/*.py``, but a crash
    inside ``transformers`` is a dependency problem, not something to patch in place.

    Args:
        lines: log lines, oldest first.

    Returns:
        ``(origin, origin_file)`` with origin in ``{"ours", "third_party", "unknown"}``.
    """
    frames = [m.group(1) for ln in lines if (m := _FRAME_RE.match(ln))]
    if not frames:
        return "unknown", ""
    deepest = frames[-1]
    norm = deepest.replace("\\", "/")
    is_ours = any(mk.replace("\\", "/") in norm for mk in _OURS_MARKERS)
    return ("ours" if is_ours else "third_party"), deepest


def classify(
    log_text: str,
    exit_code: int | None = None,
    state: str = "FAILED",
    tail_lines: int = 80,
) -> dict[str, object]:
    """Classify a job outcome into the :class:`~hygdap.bus.schema.Failure` field set.

    Args:
        log_text: the job's full stdout+stderr (or its tail).
        exit_code: process exit code, if one was captured.
        state: terminal state; ``TIMEOUT``/``STALLED``/``CANCELLED`` short-circuit the
            pattern match because their cause is known by construction.
        tail_lines: how many trailing log lines to keep in ``traceback_tail``.

    Returns:
        A dict with keys ``failure_class``, ``exception_type``, ``message``, ``origin``,
        ``origin_file``, ``traceback_tail``, ``log_line_no`` - ready to splat into
        ``Failure(**...)``.
    """
    lines = log_text.splitlines()
    exc_type, exc_msg, line_no = _last_exception(lines)
    origin, origin_file = _origin(lines)
    tail = lines[-tail_lines:]

    if state in ("TIMEOUT", "STALLED", "CANCELLED"):
        cls = {"TIMEOUT": "TIMEOUT", "STALLED": "SESSION_DEAD", "CANCELLED": "CANCELLED"}[state]
        return {
            "failure_class": cls,
            "exception_type": exc_type,
            "message": exc_msg or f"job ended in state {state}",
            "origin": origin,
            "origin_file": origin_file,
            "traceback_tail": tail,
            "log_line_no": line_no,
        }

    # Scan the whole log, not just the tail: a dependency conflict often prints its real
    # cause hundreds of lines before the traceback that finally kills the process.
    cls = "UNKNOWN"
    for name, pat in _PATTERNS:
        if pat.search(log_text):
            cls = name
            break
    if cls == "UNKNOWN" and exc_type:
        cls = "CODE_ERROR"
    if cls == "UNKNOWN" and exit_code not in (None, 0):
        cls = "CODE_ERROR"

    return {
        "failure_class": cls,
        "exception_type": exc_type,
        "message": exc_msg or (f"exit code {exit_code}" if exit_code else "unknown failure"),
        "origin": origin,
        "origin_file": origin_file,
        "traceback_tail": tail,
        "log_line_no": line_no,
    }
