"""Environment repairs that must happen before any LoRA work, shared by every payload.

Small on purpose. Each function here exists because a specific failure was observed on a
real Colab runtime, and each says which one.
"""

from __future__ import annotations

import subprocess
import sys


def neutralize_torchao() -> str | None:
    """Remove a too-old ``torchao`` so ``peft`` can dispatch LoRA layers.

    Observed 2026-09-16 (job ``20260916T0702Z-selftest-selftest-6a2c``): Colab ships
    ``torchao 0.10.0``, and ``peft 0.20.0``'s LoRA dispatcher calls
    ``is_torchao_available()``, which **raises** rather than returning ``False`` when
    torchao is installed but older than 0.16.0::

        ImportError: Found an incompatible version of torchao. Found version 0.10.0,
        but only versions above 0.16.0 are supported

    That kills ``unet.add_adapter(...)``, which is exactly what the real diffusers
    training script does - so this is not a smoke-test artifact, it would have killed the
    training run too.

    Removing torchao rather than upgrading it is deliberate. We never use torchao, and
    ``is_torchao_available()`` returns ``False`` cleanly when the package is absent.
    Upgrading it would let pip pick a torchao wheel built against a different torch, and
    rule zero for this project is: never touch the torch the Colab image ships.

    Returns:
        The version that was removed, or ``None`` if nothing needed doing.
    """
    try:
        import torchao  # noqa: F401
        from importlib.metadata import version as _version
        found = _version("torchao")
    except Exception:                                # noqa: BLE001 - absent is the good case
        return None

    try:
        major, minor, *_ = (int(x) for x in found.split(".")[:2])
        if (major, minor) >= (0, 16):
            print(f"  torchao {found} is new enough - leaving it alone", flush=True)
            return None
    except ValueError:
        pass                                          # unparseable version -> treat as old

    print(f"  torchao {found} < 0.16.0 makes peft raise on LoRA dispatch - uninstalling",
          flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
                   check=False, timeout=600)
    return found
