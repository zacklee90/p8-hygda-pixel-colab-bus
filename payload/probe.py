"""Cheapest possible job: prove the bus works end to end and report what this runtime is.

Runs in seconds, needs no GPU allocation to succeed, and writes the environment snapshot
every later job's result will be compared against. If this does not come back, nothing
else is worth submitting.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
import driveio  # noqa: E402

RUN_DIR = Path(os.environ["P8_RUN_DIR"])


def sh(cmd: list[str]) -> str:
    """Run a command and return its output, or a readable marker if it is unavailable."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (out.stdout or out.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<unavailable: {exc}>"


def main() -> int:
    """Report the runtime, write ``env_snapshot.json`` and one ``progress.json`` tick."""
    print("=== p8 bus probe ===")
    print(f"python   : {sys.version.split()[0]}  ({platform.platform()})")
    print(f"cwd      : {os.getcwd()}")
    print(f"run_dir  : {RUN_DIR}")

    total, used, free = shutil.disk_usage("/content" if Path("/content").exists() else ".")
    print(f"disk     : {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")

    smi = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
              "--format=csv,noheader"])
    print(f"nvidia-smi: {smi}")

    gpu_name, vram_gb, torch_ver, cuda_ok = None, None, None, False
    try:
        import torch
        torch_ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print(f"torch    : {torch_ver}  cuda_available={cuda_ok}  gpu={gpu_name} vram={vram_gb}GB")
    except ImportError as exc:
        print(f"torch    : NOT INSTALLED ({exc})")

    drive = Path("/content/drive/MyDrive/p8-hygda-pixel")
    print(f"drive    : mounted={drive.is_dir()}")
    if drive.is_dir():
        for name in ("hf_dataset_pixelv1_p8.zip", "default.yaml"):
            p = drive / name
            print(f"  {name:<28} {'%.1f MB' % (p.stat().st_size / 1e6) if p.is_file() else 'MISSING'}")

    snapshot = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch_ver, "cuda_available": cuda_ok, "gpu": gpu_name, "vram_gb": vram_gb,
        "nvidia_smi": smi, "disk_free_gb": round(free / 1e9, 1),
        "drive_mounted": drive.is_dir(), "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    driveio.write_json_atomic(RUN_DIR / "env_snapshot.json", snapshot)
    driveio.write_json_atomic(RUN_DIR / "progress.json",
                              {"phase": "probe", "step": None, "gpu": gpu_name,
                               "t_utc": snapshot["t_utc"]})
    print("=== probe OK ===")
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
