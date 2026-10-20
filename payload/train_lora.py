"""LoRA training job - replaces cells 2-5 of the old notebook, and the bugs that came with them.

Every design choice below is a direct answer to something that already went wrong on
2026-09-15, when three attempts produced three empty directories on Drive and no log at all:

* ``!accelerate launch`` is shell magic and **swallows the exit code**, so a crash printed
  "TRAINING COMPLETE". Here the child is launched with a real argv list through
  ``subprocess.Popen`` and the return code is the truth.
* Training output went straight onto the Drive FUSE mount, so every checkpoint crawled.
  Here ``--output_dir`` is local disk and a watcher copies finished checkpoints to Drive,
  writing a ``DONE`` marker only after the copy completes - so resume can never pick up a
  half-uploaded checkpoint.
* Nothing was ever written to disk, so a failure left no forensic trace. Here all output
  is streamed, and ``progress.json`` ticks every 60 s.
* ``RUN_NAME``/``PILOT`` were literals in a notebook cell. Here every knob is an env var.
* ``runwayml/stable-diffusion-v1-5`` was removed from the Hub. Here the id is resolved
  from a candidate list and pinned to a commit sha.
* A dependency mismatch killed the run at import. Here the lock is verified and the
  imports are smoke-tested **before** any GPU work starts, so that failure costs seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import driveio  # noqa: E402
from envfix import neutralize_torchao  # noqa: E402

REPO_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = Path(os.environ["P8_RUN_DIR"])
LOCAL_WORK = Path(os.environ.get("P8_LOCAL_WORK", "/content/p8work"))
DRIVE_ROOT = Path(os.environ.get("P8_DRIVE_BUS", "/content/drive/MyDrive/p8-hygda-pixel/bus")).parent

RUN_NAME = os.environ.get("RUN_NAME", "p8")
# P8: HFSS-verified pixel CP layouts, grid coordinates, sense in the caption (p2_build_dataset.py)
DATASET_VERSION = os.environ.get("DATASET_VERSION", "pixelv1")
PILOT = os.environ.get("PILOT", "1") == "1"
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "3000"))
CHECKPOINTING_STEPS = int(os.environ.get("CHECKPOINTING_STEPS", "500"))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
RANK = int(os.environ.get("RANK", "32"))
LR = os.environ.get("LR", "1e-4")
SEED = int(os.environ.get("SEED", "42"))
RESOLUTION = int(os.environ.get("RESOLUTION", "512"))
MIXED_PRECISION = os.environ.get("MIXED_PRECISION", "fp16")
LR_SCHEDULER = os.environ.get("LR_SCHEDULER", "cosine_with_restarts")
LR_WARMUP_STEPS = int(os.environ.get("LR_WARMUP_STEPS", "200"))
MAX_TRAIN_SAMPLES = os.environ.get("MAX_TRAIN_SAMPLES", "")
CHECKPOINTS_TOTAL_LIMIT = int(os.environ.get("CHECKPOINTS_TOTAL_LIMIT", "2"))
DIFFUSERS_TAG = os.environ.get("DIFFUSERS_TAG", "v0.30.3")

RUN_ID = os.environ.get("RUN_ID") or f"{RUN_NAME}_{'pilot' if PILOT else 'full'}_s{MAX_TRAIN_STEPS}"
OUT_LOCAL = LOCAL_WORK / "runs" / RUN_ID
OUT_DRIVE = DRIVE_ROOT / "runs" / RUN_ID

MODEL_CANDIDATES = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd-legacy/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
    "botp/stable-diffusion-v1-5",
]

_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
_LOSS_RE = re.compile(r"step_loss=([0-9.eE+-]+)")


def say(msg: str) -> None:
    """Print a marked line so it stands out in a log full of tqdm noise."""
    print(f">>> {msg}", flush=True)


def sha256_of(path: Path) -> str:
    """Streaming sha256, safe for multi-hundred-MB files."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- preflight


def verify_environment() -> dict[str, str]:
    """Check the dependency lock and import everything - before any GPU work.

    Raises:
        SystemExit: on a lock mismatch or a failed import. Both are cheap to detect here
            and expensive to discover twenty minutes into a run.
    """
    lock = REPO_DIR / "payload" / "requirements.lock"
    expected = os.environ.get("DEPS_LOCK_SHA256", "")
    if lock.is_file():
        actual = sha256_of(lock)
        say(f"lock {lock.name} sha256={actual[:12]}")
        if expected and actual != expected:
            raise SystemExit(f"requirements.lock sha256 {actual} != job's {expected}")
        specs = [ln.strip() for ln in lock.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#") and not ln.startswith("torch")]
        if specs:
            say(f"installing {len(specs)} pinned packages (torch deliberately untouched)")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *specs],
                           check=False, timeout=1800)
    else:
        say("no requirements.lock yet - run the selftest job first to mint one")

    neutralize_torchao()

    versions: dict[str, str] = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "peft", "safetensors"):
        mod = __import__(name)
        versions[name] = getattr(mod, "__version__", "?")
        say(f"  {name:<14} {versions[name]}")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GPU_NOT_AVAILABLE: torch.cuda.is_available() is False - "
                         "set Runtime > Change runtime type > GPU")
    say(f"gpu: {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    versions["gpu"] = torch.cuda.get_device_name(0)
    return versions


def resolve_model() -> tuple[str, str]:
    """Resolve the SD-1.5 mirror and its commit sha (see ``selftest.py``)."""
    from huggingface_hub import model_info

    forced = os.environ.get("MODEL_ID", "").strip()
    candidates = [forced] if forced else MODEL_CANDIDATES
    for cand in candidates:
        try:
            info = model_info(cand)
            sha = os.environ.get("MODEL_REVISION", "").strip() or getattr(info, "sha", "") or ""
            say(f"model: {cand} @ {sha[:12] or 'HEAD'}")
            return cand, sha
        except Exception as exc:                     # noqa: BLE001
            say(f"model candidate {cand} unavailable: {type(exc).__name__}")
    raise SystemExit(f"MODEL_NOT_FOUND: none of {candidates} resolved on the Hub")


def prepare_dataset() -> Path:
    """Extract the dataset zip from Drive onto local disk and sanity-check it.

    Always re-extracts: a half-extracted directory left by a killed session looks exactly
    like a good one, which cost an afternoon on 2026-09-15.

    Returns:
        Path to the ``train/`` directory holding the images and ``metadata.csv``.
    """
    zip_path = DRIVE_ROOT / f"hf_dataset_{DATASET_VERSION}_{RUN_NAME}.zip"
    if not zip_path.is_file():
        raise SystemExit(f"DATA_MISSING: {zip_path} not found on Drive")
    say(f"dataset zip: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    dest = LOCAL_WORK / "data"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        bad = [n for n in zf.namelist() if "\\" in n]
        if bad:
            raise SystemExit(f"DATA_MISSING: zip uses Windows separators ({bad[0]!r}) - "
                             f"rebuild it with python zipfile and as_posix()")
        zf.extractall(dest)

    meta = next((p for p in dest.rglob("train/metadata.csv")), None)
    if meta is None:
        listing = [str(p.relative_to(dest)) for p in list(dest.rglob("*"))[:20]]
        raise SystemExit(f"DATA_MISSING: no train/metadata.csv under {dest}; saw {listing}")
    train_dir = meta.parent
    n_png = len(list(train_dir.glob("*.png")))
    n_rows = sum(1 for _ in meta.open(encoding="utf-8")) - 1
    say(f"train dir: {train_dir}  images={n_png}  captions={n_rows}")
    if n_png != n_rows:
        raise SystemExit(f"DATA_MISSING: {n_png} images but {n_rows} captions - zip is incomplete")
    return train_dir


def fetch_training_script() -> Path:
    """Clone the diffusers example script at the tag matching the installed library."""
    import diffusers
    tag = f"v{diffusers.__version__}"
    src = LOCAL_WORK / "diffusers_src"
    script = src / "examples" / "text_to_image" / "train_text_to_image_lora.py"
    if not script.is_file():
        if src.exists():
            shutil.rmtree(src)
        say(f"cloning diffusers {tag} for the training script")
        rc = subprocess.run(["git", "clone", "--depth", "1", "--branch", tag,
                             "https://github.com/huggingface/diffusers", str(src)],
                            capture_output=True, text=True, timeout=600).returncode
        if rc != 0 or not script.is_file():
            say(f"tag {tag} unavailable, falling back to {DIFFUSERS_TAG}")
            if src.exists():
                shutil.rmtree(src)
            subprocess.run(["git", "clone", "--depth", "1", "--branch", DIFFUSERS_TAG,
                            "https://github.com/huggingface/diffusers", str(src)],
                           check=True, timeout=600)
    if not script.is_file():
        raise SystemExit(f"could not obtain {script}")
    return script


# ------------------------------------------------------------------- checkpoint sync


def _checkpoint_complete(d: Path) -> bool:
    """A diffusers checkpoint dir is complete once accelerate has written its state."""
    return d.is_dir() and any(d.glob("*.safetensors")) and (d / "random_states_0.pkl").exists()


class CheckpointWatcher(threading.Thread):
    """Copy finished local checkpoints to Drive, newest first, marking each ``DONE``.

    The ``DONE`` marker is written only after the copy returns, so resume never considers
    a checkpoint that is still uploading. Old checkpoints are pruned on both sides to keep
    the local disk from filling mid-run.
    """

    def __init__(self, local_dir: Path, drive_dir: Path, keep: int = 2, interval_s: int = 30):
        super().__init__(daemon=True)
        self.local_dir, self.drive_dir, self.keep, self.interval_s = (
            local_dir, drive_dir, keep, interval_s)
        self.stop_event = threading.Event()
        self.synced: list[str] = []

    def _sync_once(self) -> None:
        for d in sorted(self.local_dir.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[1])):
            if d.name in self.synced or not _checkpoint_complete(d):
                continue
            target = self.drive_dir / d.name
            try:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(d, target)
                (target / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                             encoding="utf-8")
                size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
                self.synced.append(d.name)
                say(f"checkpoint {d.name} -> Drive ({size / 1e6:.1f} MB), DONE written")
            except OSError as exc:
                say(f"checkpoint {d.name} sync failed ({exc}) - will retry")
                return
        for old in sorted(self.drive_dir.glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[1]))[:-self.keep]:
            shutil.rmtree(old, ignore_errors=True)

    def run(self) -> None:
        """Poll for finished checkpoints until stopped, then sync one final time."""
        while not self.stop_event.wait(self.interval_s):
            self._sync_once()
        self._sync_once()


def restore_latest_checkpoint() -> str | None:
    """Copy the newest DONE checkpoint from Drive to local disk so training can resume.

    Returns:
        ``"latest"`` when something was restored (the value ``--resume_from_checkpoint``
        expects), otherwise ``None``.
    """
    if not OUT_DRIVE.is_dir():
        return None
    done = sorted((d for d in OUT_DRIVE.glob("checkpoint-*") if (d / "DONE").exists()),
                  key=lambda p: int(p.name.split("-")[1]))
    if not done:
        return None
    newest = done[-1]
    target = OUT_LOCAL / newest.name
    if not target.exists():
        say(f"resuming from {newest.name} (copying back from Drive)")
        OUT_LOCAL.mkdir(parents=True, exist_ok=True)
        shutil.copytree(newest, target)
        (target / "DONE").unlink(missing_ok=True)
    return "latest"


# --------------------------------------------------------------------------- progress


class ProgressReporter:
    """Turn the child's tqdm stream into a ticking ``progress.json``.

    Degrades gracefully: if diffusers changes its progress-bar format the regex stops
    matching and ``step`` becomes ``None``, but the agent keeps running. Cosmetics must
    never kill a six-hour job.
    """

    def __init__(self, run_dir: Path, max_steps: int, gpu: str, interval_s: int = 60):
        self.run_dir, self.max_steps, self.gpu, self.interval_s = (
            run_dir, max_steps, gpu, interval_s)
        self.step: int | None = None
        self.loss_ema: float | None = None
        self.last_ckpt: str | None = None
        self.t0 = time.time()
        self._last_write = 0.0

    def observe(self, line: str) -> None:
        """Extract step/loss from one output line and write a tick when due."""
        if (m := _STEP_RE.search(line)):
            cur, total = int(m.group(1)), int(m.group(2))
            if total == self.max_steps or self.step is None:
                self.step = cur
        if (m := _LOSS_RE.search(line)):
            try:
                loss = float(m.group(1))
                self.loss_ema = loss if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * loss
            except ValueError:
                pass
        if "checkpoint-" in line and "Saving" in line:
            if (m := re.search(r"checkpoint-(\d+)", line)):
                self.last_ckpt = f"checkpoint-{m.group(1)}"
        if time.time() - self._last_write >= self.interval_s:
            self.write()

    def write(self) -> None:
        """Persist one progress tick (this is also the agent's liveness signal)."""
        self._last_write = time.time()
        elapsed = self._last_write - self.t0
        ips = (self.step * TRAIN_BATCH_SIZE * GRAD_ACCUM / elapsed) if self.step and elapsed else None
        eta = ((self.max_steps - self.step) * elapsed / self.step) if self.step else None
        try:
            driveio.write_json_atomic(self.run_dir / "progress.json", {
                "phase": "train", "step": self.step, "max_steps": self.max_steps,
                "loss_ema": round(self.loss_ema, 5) if self.loss_ema is not None else None,
                "img_per_s": round(ips, 2) if ips else None,
                "elapsed_s": round(elapsed), "eta_s": round(eta) if eta else None,
                "gpu": self.gpu, "last_ckpt": self.last_ckpt,
                "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        except OSError:
            pass                                      # a Drive hiccup must not stop training


# --------------------------------------------------------------------------- main


def main() -> int:
    """Run one LoRA training job end to end."""
    say(f"run_id={RUN_ID} steps={MAX_TRAIN_STEPS} ckpt_every={CHECKPOINTING_STEPS} "
        f"batch={TRAIN_BATCH_SIZE}x{GRAD_ACCUM} rank={RANK} lr={LR} seed={SEED}")
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_DRIVE.mkdir(parents=True, exist_ok=True)

    versions = verify_environment()
    model_id, revision = resolve_model()
    train_dir = prepare_dataset()
    script = fetch_training_script()

    resume = restore_latest_checkpoint() if os.environ.get("RESUME", "1") == "1" else None

    argv = [
        sys.executable, "-m", "accelerate.commands.launch", str(script),
        f"--pretrained_model_name_or_path={model_id}",
        f"--train_data_dir={train_dir}",
        "--caption_column=text",
        f"--resolution={RESOLUTION}",
        f"--train_batch_size={TRAIN_BATCH_SIZE}",
        f"--gradient_accumulation_steps={GRAD_ACCUM}",
        f"--max_train_steps={MAX_TRAIN_STEPS}",
        f"--learning_rate={LR}",
        f"--lr_scheduler={LR_SCHEDULER}",
        f"--lr_warmup_steps={LR_WARMUP_STEPS}",
        f"--output_dir={OUT_LOCAL}",                  # LOCAL disk, never the Drive mount
        f"--rank={RANK}",
        f"--mixed_precision={MIXED_PRECISION}",
        "--gradient_checkpointing",
        f"--checkpointing_steps={CHECKPOINTING_STEPS}",
        f"--checkpoints_total_limit={CHECKPOINTS_TOTAL_LIMIT}",
        f"--seed={SEED}",
        "--report_to=tensorboard",
        "--dataloader_num_workers=2",
    ]
    if revision:
        argv.append(f"--revision={revision}")
    if MAX_TRAIN_SAMPLES:
        argv.append(f"--max_train_samples={MAX_TRAIN_SAMPLES}")
    if resume:
        argv.append(f"--resume_from_checkpoint={resume}")

    say("launching: " + " ".join(argv[2:8]) + " ...")
    watcher = CheckpointWatcher(OUT_LOCAL, OUT_DRIVE, keep=CHECKPOINTS_TOTAL_LIMIT)
    watcher.start()
    reporter = ProgressReporter(RUN_DIR, MAX_TRAIN_STEPS, versions.get("gpu", "?"))
    reporter.write()

    t0 = time.time()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        reporter.observe(line)
    proc.wait()
    rc = proc.returncode
    duration = time.time() - t0

    watcher.stop_event.set()
    watcher.join(timeout=600)
    reporter.write()

    weights = OUT_LOCAL / "pytorch_lora_weights.safetensors"
    if rc != 0:
        say(f"accelerate exited {rc} after {duration:.0f}s - see the traceback above")
        return rc
    if not weights.is_file():
        say(f"accelerate exited 0 but {weights.name} is missing - refusing to call this success")
        return 1

    shutil.copy2(weights, OUT_DRIVE / weights.name)
    zip_path = DRIVE_ROOT / f"hf_dataset_{DATASET_VERSION}_{RUN_NAME}.zip"
    lock = REPO_DIR / "payload" / "requirements.lock"
    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "job_id": os.environ.get("P8_JOB_ID"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": {weights.name: sha256_of(OUT_DRIVE / weights.name)},
        "dataset": {"zip": zip_path.name, "sha256": sha256_of(zip_path)},
        "deps_lock_sha256": sha256_of(lock) if lock.is_file() else None,
        "model_id_resolved": model_id,
        "model_revision_resolved": revision,
        # The values ACTUALLY used, not the ones in the config file - an auto-applied
        # OOM fix must be visible in the artifact, or the paper's methods are fiction.
        "hyperparameters": {
            "max_train_steps": MAX_TRAIN_STEPS, "checkpointing_steps": CHECKPOINTING_STEPS,
            "train_batch_size": TRAIN_BATCH_SIZE, "gradient_accumulation_steps": GRAD_ACCUM,
            "effective_batch": TRAIN_BATCH_SIZE * GRAD_ACCUM,
            "rank": RANK, "learning_rate": LR, "lr_scheduler": LR_SCHEDULER,
            "lr_warmup_steps": LR_WARMUP_STEPS, "resolution": RESOLUTION,
            "mixed_precision": MIXED_PRECISION, "seed": SEED, "run_name": RUN_NAME,
            "pilot": PILOT, "resumed": bool(resume),
        },
        "versions": versions,
        "duration_s": round(duration),
        "final_step": reporter.step,
        "loss_ema": reporter.loss_ema,
    }
    driveio.write_json_atomic(OUT_DRIVE / "MANIFEST.json", manifest)
    say(f"DONE in {duration / 60:.1f} min -> {OUT_DRIVE}")
    say(f"weights {weights.stat().st_size / 1e6:.1f} MB, MANIFEST written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
