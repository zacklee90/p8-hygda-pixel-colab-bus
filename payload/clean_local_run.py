"""Wipe one run's LOCAL output directory, to make a session death reproducible on demand.

Killing a Colab *cell* leaves ``/content`` intact, so a retry resumes from local disk and
never exercises the Drive restore path. Killing a *session* wipes ``/content`` entirely -
and that is the case that actually matters, because a 6000-step run on a T4 takes about
12 hours and cannot finish inside one session.

Running this between a cancelled attempt and its retry turns "we hope resume-from-Drive
works" into "we watched resume-from-Drive work". It is a separate, logged job rather than
a hidden flag inside the training payload so the deletion is visible in the record.

Only ever touches ``/content`` scratch under this run's own directory; Drive is untouched,
which is the entire point - the Drive checkpoints are what the retry must find.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

LOCAL_WORK = Path(os.environ.get("P8_LOCAL_WORK", "/content/p8work"))
RUN_ID = os.environ.get("RUN_ID", "")


def main() -> int:
    """Remove ``<local work>/runs/<RUN_ID>`` and report what was there."""
    if not RUN_ID:
        print("RUN_ID is required - refusing to guess which run to clear", flush=True)
        return 1

    target = LOCAL_WORK / "runs" / RUN_ID
    resolved = target.resolve()
    if not str(resolved).startswith(str((LOCAL_WORK / "runs").resolve())):
        print(f"refusing to touch {resolved}: outside {LOCAL_WORK / 'runs'}", flush=True)
        return 1

    if not target.exists():
        print(f"{target} does not exist - nothing to clear (already a fresh session)", flush=True)
        return 0

    ckpts = sorted(p.name for p in target.glob("checkpoint-*"))
    size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
    print(f"clearing {target}", flush=True)
    print(f"  contained: {len(ckpts)} checkpoint(s) {ckpts}, {size_mb:.1f} MB", flush=True)
    shutil.rmtree(target)
    print(f"  Drive copy under runs/{RUN_ID} is untouched - the retry must restore from it",
          flush=True)
    print("=== local run directory cleared ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
