"""Mint a known-good dependency lock and resolve the SD-1.5 mirror, before spending GPU hours.

This job exists to retire one specific failure permanently. On 2026-09-15 the training run
died at import with ``ImportError: cannot import name 'FLAX_WEIGHTS_NAME'`` because the
notebook installed "the latest mutually-consistent stack" and hoped. Hope is not a pinning
strategy.

The lock produced here is a *recorded observation of a session that actually worked*:
install candidates, ``pip check``, import everything, resolve the model, run five real
training steps, then freeze. Every later job asserts the lock's sha256 and re-runs the
import smoke test, so a broken environment costs 30 seconds instead of a 20-minute
head-fake.

Rule zero: never reinstall ``torch``. Use whatever the Colab image ships. Touching torch is
the most reliable way to break a working Colab environment, and it costs several minutes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driveio  # noqa: E402
from envfix import neutralize_torchao  # noqa: E402

RUN_DIR = Path(os.environ["P8_RUN_DIR"])
REPO_DIR = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_DIR / "payload" / "requirements.lock"

# Narrow set that actually matters. torch/torchvision are deliberately absent (rule zero).
#
# Measured runtime on 2026-09-16: Colab ships Python 3.13.15 with torch 2.11.0+cu128. The
# 2024-era pins this project inherited from the prototype (transformers 4.44, accelerate
# 0.33) predate Python 3.13 entirely, so demanding them would fail before it taught us
# anything. Instead the FIRST mint resolves unpinned - and then verifies, which is the
# part the prototype skipped: pip check, import every package, and run five real training
# steps. Only a set that survives all three becomes the lock. "Install latest and hope" is
# what broke the old notebook; "install latest, prove it trains, then freeze it forever"
# is a different thing.
#
# DEP_SET lets the auto-fix loop switch strategies without a code edit.
DEP_SETS: dict[str, list[str]] = {
    "latest": [
        "diffusers", "transformers", "accelerate", "peft",
        "huggingface_hub", "safetensors", "datasets",
    ],
    "pinned_2024": [
        "diffusers==0.30.3", "transformers==4.44.2", "accelerate==0.33.0",
        "peft==0.12.0", "huggingface_hub==0.24.6", "safetensors==0.4.5",
        "datasets==2.21.0",
    ],
}
DEP_SET = os.environ.get("DEP_SET", "latest")
CANDIDATES = DEP_SETS.get(DEP_SET, DEP_SETS["latest"])

MODEL_CANDIDATES = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-legacy/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
    "botp/stable-diffusion-v1-5",
]

IMPORTS = ["torch", "diffusers", "transformers", "accelerate", "peft",
           "huggingface_hub", "safetensors", "datasets"]


def run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Run a command, echoing it and its output into the job log."""
    print(f"$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    if p.stdout:
        print(p.stdout[-8000:], flush=True)
    if p.stderr:
        print(p.stderr[-4000:], flush=True)
    return p


def install(specs: list[str]) -> None:
    """Install the candidate set without touching torch."""
    run([sys.executable, "-m", "pip", "install", "-q", *specs])


def smoke_imports() -> dict[str, str]:
    """Import every package that matters and report its version.

    Raises:
        SystemExit: if any import fails - there is no point resolving models or
            allocating GPU memory in a broken environment.
    """
    versions: dict[str, str] = {}
    for name in IMPORTS:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "?")
            print(f"  import {name:<16} {versions[name]}", flush=True)
        except Exception as exc:                     # noqa: BLE001 - report, do not mask
            print(f"  import {name:<16} FAILED: {type(exc).__name__}: {exc}", flush=True)
            raise
    return versions


def resolve_model() -> tuple[str, str]:
    """Find the first reachable SD-1.5 mirror and pin it to a commit sha.

    The original ``runwayml`` repo was removed from the Hub, so the id must be discovered
    rather than assumed. Recording the commit sha means a re-run is reproducible even if
    the mirror list changes again.

    Returns:
        ``(model_id, commit_sha)``.

    Raises:
        SystemExit: when no candidate resolves.
    """
    from huggingface_hub import model_info

    for cand in MODEL_CANDIDATES:
        try:
            info = model_info(cand)
            sha = getattr(info, "sha", "") or ""
            print(f"  MODEL OK   {cand}  sha={sha[:12]}", flush=True)
            return cand, sha
        except Exception as exc:                     # noqa: BLE001
            print(f"  MODEL miss {cand}: {type(exc).__name__}", flush=True)
    raise SystemExit(f"no SD-1.5 mirror resolved from {MODEL_CANDIDATES}")


def five_training_steps(model_id: str, revision: str) -> float:
    """Run five real LoRA optimizer steps to prove the stack trains, not merely imports.

    This deliberately mirrors what the real job does, because the first version of this
    check did not and produced a misleading pass. It trained the FULL UNet with fp16
    master weights and no gradient scaler; loss went to ``nan`` on step 2 while the job
    still reported success, and the images/second it measured described full-UNet training
    rather than LoRA. Both numbers were useless.

    What accelerate actually does for ``mixed_precision="fp16"`` - and therefore what this
    now does - is keep fp32 master weights, run the forward under ``autocast``, and scale
    the loss. Only the LoRA adapters are trainable.

    Args:
        model_id: resolved base model.
        revision: commit sha to pin.

    Returns:
        Measured images/second for LoRA training, used to recompute the CU budget.

    Raises:
        SystemExit: if the loss is not finite - a stack that produces ``nan`` in five
            steps has not been certified, whatever the exit code says.
    """
    import math

    import torch
    import torch.nn.functional as F
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank = int(os.environ.get("RANK", "32"))
    alpha = int(os.environ.get("LORA_ALPHA", "16"))
    batch = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", revision=revision or None, torch_dtype=torch.float32
    ).to(device)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    ))
    unet.enable_gradient_checkpointing()
    unet.train()

    params = [p for p in unet.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"  trainable {n_train / 1e6:.2f} M of {n_total / 1e6:.1f} M "
          f"({100 * n_train / n_total:.2f} %), rank={rank} alpha={alpha}", flush=True)

    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    latents = torch.randn(batch, 4, 64, 64, device=device)
    emb = torch.randn(batch, 77, 768, device=device)
    target = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (batch,), device=device).long()

    t0 = time.time()
    for step in range(5):
        with torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
            pred = unet(latents, timesteps, encoder_hidden_states=emb).sample
        loss = F.mse_loss(pred.float(), target.float())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        value = loss.item()
        print(f"  step {step + 1}/5 loss={value:.4f}", flush=True)
        if not math.isfinite(value):
            raise SystemExit(
                f"loss became {value} at step {step + 1} - refusing to certify a stack "
                f"that diverges in five steps"
            )
    dt = time.time() - t0
    ips = (5 * batch) / dt
    print(f"  5 LoRA steps in {dt:.1f}s -> {ips:.2f} img/s on {device}", flush=True)
    return ips


def main() -> int:
    """Install, verify, resolve, train five steps, and emit the lock."""
    print(f"=== P8 selftest: mint a dependency lock by observation (DEP_SET={DEP_SET}) ===",
          flush=True)
    print(f"python {sys.version.split()[0]}", flush=True)
    install(CANDIDATES)

    print("\n--- environment repairs ---", flush=True)
    removed = neutralize_torchao()

    print("\n--- pip check ---", flush=True)
    check = run([sys.executable, "-m", "pip", "check"])
    # Colab's base image nearly always has some unrelated conflict (it ships hundreds of
    # packages). Failing on any of them would reject a perfectly good environment, so only
    # a conflict naming one of OUR packages counts.
    ours = {spec.split("==")[0].strip().lower().replace("_", "-") for spec in CANDIDATES}
    ours |= {"torch", "torchvision"}
    relevant = [ln for ln in (check.stdout or "").splitlines()
                if any(name in ln.lower() for name in ours)]
    if relevant:
        print("pip check conflicts involving our packages - refusing to certify:", flush=True)
        for ln in relevant:
            print(f"  {ln}", flush=True)
        return 1
    if check.returncode != 0:
        print("pip check reported conflicts, none involving our packages - continuing",
              flush=True)

    print("\n--- import smoke test ---", flush=True)
    versions = smoke_imports()

    print("\n--- resolve SD-1.5 mirror ---", flush=True)
    model_id, sha = resolve_model()

    print("\n--- five real training steps ---", flush=True)
    ips = five_training_steps(model_id, sha)

    print("\n--- pip freeze ---", flush=True)
    freeze = run([sys.executable, "-m", "pip", "freeze"]).stdout
    keep = {spec.split("==")[0].strip().lower().replace("_", "-") for spec in CANDIDATES}
    keep |= {"torch", "torchvision", "numpy", "bitsandbytes"}
    lock_lines = sorted(
        ln.strip() for ln in freeze.splitlines()
        if "==" in ln and ln.split("==")[0].strip().lower().replace("_", "-") in keep
    )

    if len(lock_lines) < len(CANDIDATES):
        print(f"freeze produced only {len(lock_lines)} pinned lines for {len(CANDIDATES)} "
              f"packages - refusing to certify an incomplete lock", flush=True)
        return 1

    import torch
    snapshot = {
        "python": sys.version.split()[0],
        "versions": versions,
        "model_id_resolved": model_id,
        "model_revision_resolved": sha,
        "img_per_s_5step": round(ips, 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "lock_lines": lock_lines,
        "torchao_removed": removed,
        "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    driveio.write_json_atomic(RUN_DIR / "env_snapshot.json", snapshot)
    # The lock is written to Drive, not into the checkout: the local driver commits it, so
    # the lock in the repo is always one that a real session produced and a human can see.
    (RUN_DIR / "requirements.lock").write_text("\n".join(lock_lines) + "\n", encoding="utf-8")

    print("\n=== LOCK ===", flush=True)
    print("\n".join(lock_lines), flush=True)
    print(f"\nmodel_id={model_id}\nrevision={sha}\nimg_per_s={ips:.2f}", flush=True)
    print("=== selftest OK ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
