"""Colab entry point. Lives on Drive so the notebook cell never has to change.

This is the ONLY file that knows it is running on Colab. Everything else
(``agent/``, ``payload/``) is plain Python that would run on any GPU host, which is what
makes a later move to Colab Enterprise or a rented GPU a matter of changing who launches
the supervisor.

Deployed to ``H:\\My Drive\\p8-hygda-pixel\\bootstrap.py`` by
``scripts/p2_colab_bus.py bootstrap-refresh``. The notebook cell is::

    from google.colab import drive; drive.mount('/content/drive')
    exec(open('/content/drive/MyDrive/p8-hygda-pixel/bootstrap.py').read())
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/zacklee90/p8-hygda-pixel-colab-bus"
BRANCH = "main"
CHECKOUT = Path("/content/bus")
DRIVE_ROOT = Path("/content/drive/MyDrive/p8-hygda-pixel")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def main() -> int:
    """Clone or refresh the control repo, then hand off to the supervisor."""
    if not DRIVE_ROOT.is_dir():
        print(f"FATAL: {DRIVE_ROOT} not found. Did the Drive mount above succeed, and is "
              f"this the tuantu90@hanyang.ac.kr account?", flush=True)
        return 70

    if (CHECKOUT / ".git").is_dir():
        _run(["git", "-C", str(CHECKOUT), "fetch", "--depth", "1", "origin", BRANCH])
        _run(["git", "-C", str(CHECKOUT), "reset", "--hard", f"origin/{BRANCH}"])
    else:
        _run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(CHECKOUT)])

    if not (CHECKOUT / "agent" / "supervisor.py").is_file():
        print("FATAL: checkout looks wrong - agent/supervisor.py is missing", flush=True)
        return 70

    os.environ.setdefault("P8_DRIVE_BUS", str(DRIVE_ROOT / "bus"))
    os.environ.setdefault("P8_LOCAL_WORK", "/content/p8work")
    os.environ.setdefault("P8_SESSION_ID", f"colab-{int(time.time())}")
    os.environ.setdefault("P8_POLL_S", "15")
    os.environ["PYTHONUNBUFFERED"] = "1"

    print(f"bus       : {os.environ['P8_DRIVE_BUS']}", flush=True)
    print(f"session   : {os.environ['P8_SESSION_ID']}", flush=True)
    print("handing off to the supervisor - leave this cell running.", flush=True)

    return subprocess.run(
        [sys.executable, "-u", str(CHECKOUT / "agent" / "supervisor.py")],
        cwd=str(CHECKOUT), env=dict(os.environ),
    ).returncode


if __name__ == "__main__" or True:   # exec()'d from a notebook cell, so __name__ is not __main__
    _rc = main()
    print(f"[bootstrap] supervisor exited rc={_rc}", flush=True)
