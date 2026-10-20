"""Keep the Colab agent alive and up to date for the lifetime of the runtime.

This is the piece that makes the loop autonomous. It re-pulls the control repo and
relaunches the agent whenever the agent's own code changes, and restarts it after a
crash - so a fix I push mid-session takes effect without anyone touching the browser.

Exit code contract with ``colab_agent.py``::

    75  agent updated itself -> pull and relaunch immediately
    70  fatal -> stop, a human is needed
    0   clean shutdown
    *   crash -> pull and relaunch after a short backoff
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
AGENT = REPO_DIR / "agent" / "colab_agent.py"
BACKOFF_S = 10
MAX_CONSECUTIVE_CRASHES = 10


def git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git inside the control-repo checkout."""
    return subprocess.run(["git", "-C", str(REPO_DIR), *args],
                          capture_output=True, text=True, timeout=120)


def pull() -> str:
    """Hard-reset the checkout to origin/main and return the new HEAD."""
    git("fetch", "--depth", "1", "origin", "main")
    git("reset", "--hard", "origin/main")
    return git("rev-parse", "HEAD").stdout.strip()


def main() -> int:
    """Supervise the agent until the runtime dies or a fatal condition is reported."""
    crashes = 0
    while True:
        head = pull()
        print(f"[supervisor] launching agent at {head[:12]}", flush=True)
        rc = subprocess.run([sys.executable, "-u", str(AGENT)], cwd=str(REPO_DIR),
                            env=dict(os.environ)).returncode

        if rc == 75:
            print("[supervisor] agent requested a restart (its code changed)", flush=True)
            crashes = 0
            continue
        if rc == 70:
            print("[supervisor] agent reported a FATAL condition - stopping. "
                  "A human needs to look at the last result.json.", flush=True)
            return 70
        if rc == 0:
            print("[supervisor] agent exited cleanly", flush=True)
            return 0

        crashes += 1
        print(f"[supervisor] agent crashed (rc={rc}), consecutive={crashes}", flush=True)
        if crashes >= MAX_CONSECUTIVE_CRASHES:
            print("[supervisor] too many consecutive crashes - stopping rather than "
                  "spinning. Check the control repo for a bad agent commit.", flush=True)
            return 70
        time.sleep(BACKOFF_S)


if __name__ == "__main__":
    sys.exit(main())
