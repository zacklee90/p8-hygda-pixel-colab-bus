"""Colab-side worker: poll the control repo for jobs, run them, stream results to Drive.

Runs inside one long-lived Colab cell, under ``supervisor.py``. Contains NOTHING specific
to Colab - no ``google.colab`` import, no OAuth - so the same worker can later be launched
by Colab Enterprise (``gcloud colab executions``) or any GPU host without modification.
Everything Colab-specific lives in ``bootstrap.py`` on Drive.

Exit codes read by the supervisor::

    0   clean shutdown (nothing left to do)
    70  fatal - a human must intervene, do not relaunch
    75  the agent's own code changed; re-pull and relaunch me

Design notes that are load-bearing:

* **The log is written as immutable numbered chunks**, never appended to one file: a FUSE
  mount re-uploads the whole file on every flush, which is what makes long runs crawl.
* **Liveness is "something under my run dir was touched recently"** - ``progress.json``
  ticks every 60 s, so there is no separate heartbeat to go stale on its own.
* **A result is written on every path**, including timeout, cancellation and crash. The
  three failed attempts that motivated this whole design left zero forensic trace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
REPO_DIR = AGENT_DIR.parent
sys.path.insert(0, str(AGENT_DIR))

import classify as classify_mod  # noqa: E402
import driveio  # noqa: E402
from schema import Failure, Job, JobResult, utc_now  # noqa: E402

EXIT_CLEAN = 0
EXIT_FATAL = 70
EXIT_RESTART = 75

DRIVE_BUS = Path(os.environ.get("P8_DRIVE_BUS", "/content/drive/MyDrive/p8-hygda-pixel/bus"))
LOCAL_WORK = Path(os.environ.get("P8_LOCAL_WORK", "/content/p8work"))
POLL_S = int(os.environ.get("P8_POLL_S", "15"))
IDLE_EXIT_S = int(os.environ.get("P8_IDLE_EXIT_S", "0"))  # 0 = never exit on idle


def log(msg: str) -> None:
    """One compact line to the cell. Colab throttles output-heavy cells, so stay terse."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command inside the control-repo checkout."""
    return subprocess.run(["git", "-C", str(REPO_DIR), *args],
                          capture_output=True, text=True, timeout=120)


def head_sha() -> str:
    """Current HEAD of the checkout (recorded into every result)."""
    return git("rev-parse", "HEAD").stdout.strip()


def agent_fingerprint() -> str:
    """Hash of the agent's own source, so it can notice when it has been updated."""
    import hashlib
    h = hashlib.sha1()
    for name in sorted(p.name for p in AGENT_DIR.glob("*.py")):
        h.update((AGENT_DIR / name).read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- claims


def claim(job: Job) -> bool:
    """Take ownership of a job, unless a live session already holds it.

    A claim whose heartbeat has gone cold belongs to a session that died; if the job is
    resumable we take it over, otherwise we mark it abandoned rather than silently
    repeating work that may have had side effects.

    Args:
        job: the job being considered.

    Returns:
        True if this session may run the job.
    """
    run_dir = DRIVE_BUS / "runs" / job.job_id
    claim_path = run_dir / "claim.json"
    existing = driveio.read_json(claim_path)
    if isinstance(existing, dict):
        age = time.time() - float(existing.get("t_heartbeat", 0))
        if age < 120:
            return False
        if not job.resume:
            abandoned = JobResult(
                job_id=job.job_id, idempotency_key=job.idempotency_key, attempt=job.attempt,
                state="ABANDONED", started_utc=utc_now(), ended_utc=utc_now(),
                failure=Failure(
                    failure_class="SESSION_DEAD",
                    message=f"stale claim ({age:.0f}s old) and resume=false - not repeating "
                            f"work whose side effects are unknown",
                ).__dict__,
            )
            driveio.write_json_atomic(run_dir / "result.json", json.loads(abandoned.to_json()))
            return False
        log(f"taking over stale claim on {job.job_id} (age {age:.0f}s)")
    driveio.write_json_atomic(claim_path, {
        "session": os.environ.get("P8_SESSION_ID", "unknown"),
        "t_claimed": time.time(), "t_heartbeat": time.time(), "claimed_utc": utc_now(),
    })
    return True


def heartbeat(job_id: str) -> None:
    """Refresh this session's claim timestamp."""
    path = DRIVE_BUS / "runs" / job_id / "claim.json"
    data = driveio.read_json(path) or {}
    data["t_heartbeat"] = time.time()
    driveio.write_json_atomic(path, data)


# --------------------------------------------------------------------------- log pump


class LogPump:
    """Drain a child process's output into local disk and into Drive chunks.

    The local file is authoritative and complete; the Drive chunks are what the operator
    watches. Flushes fast for the first few minutes (where essentially every failure of
    this pipeline has happened) and slowly afterwards, so a 12-hour run does not produce
    thousands of tiny files.
    """

    def __init__(self, job_id: str, log_dir: Path, local_log: Path,
                 chunk_bytes: int = 8192, fast_s: int = 10, slow_s: int = 60,
                 fast_phase_s: int = 300):
        self.log_dir = log_dir
        self.local_log = local_log
        self.chunk_bytes = chunk_bytes
        self.fast_s, self.slow_s, self.fast_phase_s = fast_s, slow_s, fast_phase_s
        self.job_id = job_id
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lines_seen = 0
        local_log.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(local_log, "a", encoding="utf-8", errors="replace", newline="\n")

    def write(self, line: str) -> None:
        """Record one output line."""
        self._fh.write(line)
        self.lines_seen += 1
        with self._lock:
            self._buf.append(line)

    def _flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            payload, self._buf = "".join(self._buf), []
        self._seq += 1
        try:
            driveio.write_chunk(self.log_dir, self._seq, payload)
        except OSError as exc:                      # Drive hiccup must not kill training
            log(f"chunk {self._seq} failed to write: {exc}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            interval = self.fast_s if (time.time() - self._t0) < self.fast_phase_s else self.slow_s
            self._stop.wait(interval)
            self._fh.flush()
            self._flush()

    def start(self) -> "LogPump":
        """Begin the background flusher."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def close(self) -> int:
        """Stop flushing, write the tail, copy the complete log to Drive.

        Returns:
            Number of chunks written.
        """
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=30)
        self._fh.flush()
        self._flush()
        self._fh.close()
        try:
            shutil.copy2(self.local_log, self.log_dir.parent / "full.log")
        except OSError as exc:
            log(f"could not copy full.log: {exc}")
        return self._seq


# --------------------------------------------------------------------------- run


def _cancel_requested(job_id: str) -> bool:
    """True once a ``<job_id>.cancel`` marker has been pushed to the control repo."""
    git("fetch", "--depth", "1", "origin", "main")
    out = git("ls-tree", "-r", "--name-only", "origin/main", "jobs/")
    return f"jobs/{job_id}.cancel" in out.stdout.splitlines()


def run_job(job: Job) -> JobResult:
    """Execute one job end to end and return its structured result.

    The child is launched with ``subprocess.Popen(argv_list)`` - a real argv, no shell,
    no string interpolation - so the exit code is real and a caption containing a quote
    cannot corrupt the command line.
    """
    run_dir = DRIVE_BUS / "runs" / job.job_id
    log_dir = run_dir / "log"
    local_log = LOCAL_WORK / f"{job.job_id}.log"
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)

    started = utc_now()
    t0 = time.time()
    env = dict(os.environ)
    env.update(job.env)
    env["P8_JOB_ID"] = job.job_id
    env["P8_RUN_DIR"] = str(run_dir)
    env["P8_LOCAL_WORK"] = str(LOCAL_WORK)
    env["P8_DRIVE_BUS"] = str(DRIVE_BUS)
    env["PYTHONUNBUFFERED"] = "1"

    argv = [sys.executable, "-u", str(REPO_DIR / job.payload_entry), *job.payload_args]
    log(f"running {job.job_id} -> {' '.join(argv[-2:])}")

    pump = LogPump(job.job_id, log_dir, local_log).start()
    pump.write(f"# job {job.job_id} attempt {job.attempt} commit {head_sha()[:12]}\n")
    pump.write(f"# argv: {argv}\n# env overrides: {json.dumps(job.env, sort_keys=True)}\n")

    state, exit_code, sig = "SUCCEEDED", None, None
    proc = subprocess.Popen(argv, cwd=str(REPO_DIR), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="replace")

    def watchdog() -> None:
        """Enforce timeout and honour a pushed cancel marker."""
        nonlocal state
        last_cancel_check = 0.0
        while proc.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - t0 > job.timeout_s:
                state = "TIMEOUT"
                log(f"timeout after {job.timeout_s}s - terminating")
                proc.terminate()
                return
            if now - last_cancel_check > 30:
                last_cancel_check = now
                try:
                    if _cancel_requested(job.job_id):
                        state = "CANCELLED"
                        log("cancel marker seen - terminating")
                        proc.terminate()
                        return
                except (OSError, subprocess.SubprocessError):
                    pass
            if int(now - t0) % 60 < 5:
                try:
                    heartbeat(job.job_id)
                except OSError:
                    pass

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        pump.write(line)
    proc.wait()
    exit_code = proc.returncode
    if exit_code is not None and exit_code < 0:
        sig, exit_code = -exit_code, None
    if state == "SUCCEEDED" and (exit_code not in (0, None) or sig):
        state = "FAILED"

    n_chunks = pump.close()
    duration = time.time() - t0
    log_text = local_log.read_text(encoding="utf-8", errors="replace")

    failure = None
    if state != "SUCCEEDED":
        failure = classify_mod.classify(log_text, exit_code=exit_code, state=state)
        failure = Failure(**failure).__dict__

    progress = driveio.read_json(run_dir / "progress.json") or {}
    result = JobResult(
        job_id=job.job_id, idempotency_key=job.idempotency_key, attempt=job.attempt,
        state=state, exit_code=exit_code, signal=sig,
        started_utc=started, ended_utc=utc_now(), duration_s=round(duration, 1),
        repo_commit=head_sha(),
        failure=failure,
        metrics={"last_step": progress.get("step"), "img_per_s": progress.get("img_per_s"),
                 "gpu": progress.get("gpu"), "loss_ema": progress.get("loss_ema")},
        artifacts=sorted(p.name for p in run_dir.glob("*") if p.is_file()),
        env_snapshot=driveio.read_json(run_dir / "env_snapshot.json") or {},
        log={"chunks": n_chunks, "lines": pump.lines_seen, "full_log": "full.log"},
    )
    driveio.write_json_atomic(run_dir / "result.json", json.loads(result.to_json()))
    log(f"{job.job_id} -> {state} in {duration:.0f}s ({n_chunks} chunks)")
    return result


# --------------------------------------------------------------------------- main loop


def pending_jobs() -> list[Job]:
    """Job files present in the checkout that have no result yet, oldest first."""
    out: list[Job] = []
    jobs_dir = REPO_DIR / "jobs"
    if not jobs_dir.is_dir():
        return out
    for p in sorted(jobs_dir.glob("*.json")):
        try:
            job = Job.from_json(p.read_text(encoding="utf-8"))
            job.validate()
        except (ValueError, OSError) as exc:
            log(f"skipping malformed {p.name}: {exc}")
            continue
        if (DRIVE_BUS / "runs" / job.job_id / "result.json").exists():
            continue
        out.append(job)
    return out


def main() -> int:
    """Poll for work until the runtime dies or the agent's own code changes."""
    os.environ.setdefault("P8_SESSION_ID", f"colab-{int(time.time())}")
    (DRIVE_BUS / "runs").mkdir(parents=True, exist_ok=True)
    fingerprint = agent_fingerprint()
    log(f"agent up: repo={head_sha()[:12]} bus={DRIVE_BUS} poll={POLL_S}s")

    last_head = head_sha()
    idle_since = time.time()
    while True:
        for job in pending_jobs():
            if not claim(job):
                continue
            idle_since = time.time()
            try:
                run_job(job)
            except Exception:                        # noqa: BLE001 - must never kill the loop
                tb = traceback.format_exc()
                log("AGENT CRASH while running a job:\n" + tb)
                driveio.write_json_atomic(
                    DRIVE_BUS / "runs" / job.job_id / "result.json",
                    json.loads(JobResult(
                        job_id=job.job_id, idempotency_key=job.idempotency_key,
                        attempt=job.attempt, state="FAILED", started_utc=utc_now(),
                        ended_utc=utc_now(), repo_commit=head_sha(),
                        failure=Failure(**classify_mod.classify(tb, exit_code=1)).__dict__,
                    ).to_json()))

        git("fetch", "--depth", "1", "origin", "main")
        remote = git("rev-parse", "origin/main").stdout.strip()
        if remote and remote != last_head:
            log(f"new commit {remote[:12]} - updating checkout")
            git("reset", "--hard", "origin/main")
            last_head = head_sha()
            if agent_fingerprint() != fingerprint:
                log("agent code changed - asking supervisor to relaunch me")
                return EXIT_RESTART
            continue

        if IDLE_EXIT_S and (time.time() - idle_since) > IDLE_EXIT_S:
            log("idle timeout - shutting down cleanly")
            return EXIT_CLEAN
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
