"""Job / result schema shared by the local driver and the Colab agent.

STDLIB ONLY. This module is mirrored verbatim into ``colab-bus/agent/schema.py`` so the
Colab side gets it through the control repo without installing anything; the program
convention is to copy modules rather than extract a shared library (umbrella CLAUDE.md),
and ``tests/test_bus_mirror.py`` asserts the two copies stay byte-identical.

A note on binding jobs to code: ``repo_commit`` is an OUTPUT, recorded in the result at
run time, not an input stamped into the job. The job file and any code fix are pushed in
ONE commit, so "this job exists in HEAD" is itself the binding - a fix always arrives with
a new job_id, and the agent never re-runs a job_id it has already claimed.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1

KINDS = ("probe", "selftest", "train_lora", "shell", "python")

#: Terminal states written into ``result.json``.
STATES = ("SUCCEEDED", "FAILED", "TIMEOUT", "STALLED", "CANCELLED", "ABANDONED")

#: Failure classes the auto-fix policy dispatches on (``docs/colab_autonomy.md``).
FAILURE_CLASSES = (
    "OOM_CUDA",
    "MODEL_NOT_FOUND",
    "DEPENDENCY_CONFLICT",
    "DATA_MISSING",
    "DRIVE_IO",
    "DISK_FULL",
    "GPU_ABSENT",
    "USAGE_LIMIT",
    "SESSION_DEAD",
    "TIMEOUT",
    "CANCELLED",
    "CODE_ERROR",
    "UNKNOWN",
)


def utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ`` (second resolution, sorts lexically)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_job_id(kind: str, label: str) -> str:
    """Build a sortable, human-readable, collision-free job id.

    Args:
        kind: one of :data:`KINDS`.
        label: short slug describing the work, e.g. ``"clean-pilot"``.

    Returns:
        ``<utc compact>-<kind>-<label>-<4 hex>``, e.g.
        ``20260915T2340Z-train_lora-clean-pilot-9f2c``.
    """
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%MZ")
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "job"
    return f"{stamp}-{kind}-{slug}-{uuid.uuid4().hex[:4]}"


def normalize_message(msg: str) -> str:
    """Strip everything volatile from an error message so two runs of the same bug match.

    Removes absolute paths, line numbers, hex addresses, digits runs and timestamps -
    the things that differ between two occurrences of one underlying failure.

    Args:
        msg: raw exception message or log line.

    Returns:
        A normalized string suitable for hashing into a failure signature.
    """
    s = msg.strip()
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "PATH", s)          # windows paths
    s = re.sub(r"/(?:[\w.+-]+/)+[\w.+-]+", "PATH", s)        # posix paths
    s = re.sub(r"line \d+", "line N", s)
    s = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*", "TS", s)
    s = re.sub(r"\d+", "N", s)
    return re.sub(r"\s+", " ", s).strip()


def failure_signature(failure_class: str, exception_type: str, message: str) -> str:
    """sha1 over (class, exception type, normalized message) - the anti-loop primitive.

    Two attempts sharing a signature mean the fix applied between them changed nothing.

    Args:
        failure_class: one of :data:`FAILURE_CLASSES`.
        exception_type: e.g. ``"ImportError"``; ``""`` when there was no Python exception.
        message: raw message; normalized internally.

    Returns:
        40-char lowercase hex sha1.
    """
    payload = f"{failure_class}\x00{exception_type}\x00{normalize_message(message)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class Job:
    """One unit of work handed to the Colab agent."""

    job_id: str
    kind: str
    idempotency_key: str
    payload_entry: str
    schema_version: int = SCHEMA_VERSION
    attempt: int = 1
    parent_job_id: str | None = None
    fix_note: str | None = None
    #: The fix carried by THIS attempt, and the failure signature it answers. Recorded at
    #: submit time - not derived from the outcome, which is a different thing entirely:
    #: the failure class an attempt PRODUCES says nothing about the fix it CARRIED.
    fix_class: str | None = None
    responding_to: str | None = None
    created_utc: str = field(default_factory=utc_now)
    timeout_s: int = 43200
    stall_s: int = 900
    resume: bool = True
    requires: dict[str, Any] = field(default_factory=dict)
    deps_lock_sha256: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    payload_args: list[str] = field(default_factory=list)
    drive_run_dir: str = ""

    def validate(self) -> None:
        """Raise ``ValueError`` if this job could not possibly be executed.

        Checked here rather than on the Colab side so a malformed job never costs a
        round trip (or GPU seconds).
        """
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {self.schema_version} != {SCHEMA_VERSION}")
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {KINDS}")
        if not self.job_id or "/" in self.job_id or "\\" in self.job_id:
            raise ValueError(f"bad job_id {self.job_id!r}")
        if not self.idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        if not self.payload_entry.startswith("payload/"):
            raise ValueError(f"payload_entry must live under payload/, got {self.payload_entry!r}")
        if self.attempt < 1:
            raise ValueError("attempt starts at 1")
        if self.timeout_s <= 0 or self.stall_s <= 0:
            raise ValueError("timeout_s and stall_s must be positive")
        for k, v in self.env.items():
            if not isinstance(v, str):
                raise ValueError(f"env[{k!r}] must be a string, got {type(v).__name__}")

    def to_json(self) -> str:
        """Serialize to the on-disk job JSON (stable key order, trailing newline)."""
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "Job":
        """Parse a job JSON, ignoring unknown keys so an older agent tolerates new fields."""
        raw = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Failure:
    """Structured cause of a non-success terminal state."""

    failure_class: str
    exception_type: str = ""
    message: str = ""
    origin: str = "unknown"          # "ours" | "third_party" | "unknown"
    origin_file: str = ""
    traceback_tail: list[str] = field(default_factory=list)
    log_line_no: int | None = None
    signature_sha1: str = ""

    def __post_init__(self) -> None:
        if not self.signature_sha1:
            self.signature_sha1 = failure_signature(
                self.failure_class, self.exception_type, self.message
            )


@dataclass
class JobResult:
    """What the agent writes to ``runs/<job_id>/result.json`` - always, on every path."""

    job_id: str
    idempotency_key: str
    state: str
    attempt: int = 1
    schema_version: int = SCHEMA_VERSION
    exit_code: int | None = None
    signal: int | None = None
    started_utc: str = ""
    ended_utc: str = ""
    duration_s: float = 0.0
    repo_commit: str = ""            # HEAD the agent actually ran at - recorded, not demanded
    failure: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    env_snapshot: dict[str, Any] = field(default_factory=dict)
    log: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to ``result.json`` (stable key order, trailing newline)."""
        return json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "JobResult":
        """Parse a result JSON, ignoring unknown keys."""
        raw = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    @property
    def signature(self) -> str:
        """Failure signature, or ``""`` when the job succeeded."""
        return (self.failure or {}).get("signature_sha1", "")
