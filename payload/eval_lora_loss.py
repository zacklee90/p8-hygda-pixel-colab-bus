"""Denoising loss of LoRA checkpoints on the train and validation splits (bus job ``kind: python``, GPU).

The training loop only logs a noisy per-batch loss. To compare train and validation on equal terms, every checkpoint
is scored on the SAME images, noise draws and timesteps:

* train subset: ``N_TRAIN`` images drawn once (seed 42) from ``train/`` of the dataset zip;
* validation:   every image of ``val/`` (held out by design_id, never seen in training);
* per image ``N_DRAWS`` (timestep, noise) pairs from a fixed generator; t ~ U{0..999} as in training;
* loss = MSE(eps_pred, eps), the objective of diffusers' ``train_text_to_image_lora.py`` (epsilon prediction,
  no SNR weighting, no noise offset), latents = VAE sample x scaling factor, text = CLIP embedding of the caption.

Inputs (env):
    RUN_ID        run folder under MyDrive/p8-hygda-pixel/runs/ (its checkpoint-*/ and final weights are scored)
    DATASET_ZIP   default MyDrive/p8-hygda-pixel/hf_dataset_pixelv1_p8.zip
    N_TRAIN, N_DRAWS, BATCH   default 128 / 4 / 8
    INCLUDE_BASE  "1" (default) also scores the base model without LoRA (step 0)

Writes ``MyDrive/p8-hygda-pixel/eval/<RUN_ID>/lora_loss.json`` after every checkpoint (resumable: steps already in the
file are skipped).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import zipfile
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("P8_DRIVE_BUS", "/content/drive/MyDrive/p8-hygda-pixel/bus")).parent
RUN_ID = os.environ["RUN_ID"]
DATASET_ZIP = Path(os.environ.get("DATASET_ZIP", str(DRIVE_ROOT / "hf_dataset_pixelv1_p8.zip")))
N_TRAIN = int(os.environ.get("N_TRAIN", "128"))
N_DRAWS = int(os.environ.get("N_DRAWS", "4"))
BATCH = int(os.environ.get("BATCH", "8"))
INCLUDE_BASE = os.environ.get("INCLUDE_BASE", "1") == "1"
BASE = os.environ.get("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
REVISION = os.environ.get("MODEL_REVISION", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14") or None
LOCAL = Path(os.environ.get("P8_LOCAL_WORK", "/content/p8work")) / "lossdata"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envfix import neutralize_torchao  # noqa: E402


def _split(name: str) -> list[tuple[Path, str]]:
    meta = next(LOCAL.rglob(f"{name}/metadata.csv"))
    return [(meta.parent / r["file_name"], r["text"]) for r in csv.DictReader(open(meta, encoding="utf-8"))]


def main() -> int:
    neutralize_torchao()
    import numpy as np
    import torch
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from PIL import Image

    run_dir = DRIVE_ROOT / "runs" / RUN_ID
    if not run_dir.is_dir() or not DATASET_ZIP.is_file():
        print(f"DATA_MISSING: {run_dir} or {DATASET_ZIP}", flush=True)
        return 1
    out_dir = DRIVE_ROOT / "eval" / RUN_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lora_loss.json"
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {"rows": []}

    if not LOCAL.exists():
        LOCAL.mkdir(parents=True)
        with zipfile.ZipFile(DATASET_ZIP) as zf:
            zf.extractall(LOCAL)
    rng = np.random.default_rng(42)
    train = _split("train")
    train = [train[i] for i in sorted(rng.choice(len(train), size=min(N_TRAIN, len(train)), replace=False))]
    sets = {"train": train, "val": _split("val")}
    print({k: len(v) for k, v in sets.items()}, flush=True)

    ckpts: list[tuple[int, Path | None]] = [(0, None)] if INCLUDE_BASE else []
    for d in sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
        if (d / "pytorch_lora_weights.safetensors").is_file():
            ckpts.append((int(d.name.split("-")[1]), d))
    if (run_dir / "pytorch_lora_weights.safetensors").is_file():
        man = run_dir / "MANIFEST.json"
        final = json.loads(man.read_text(encoding="utf-8")).get("final_step") if man.is_file() else None
        if final is not None and all(s != final for s, _ in ckpts):
            ckpts.append((int(final), run_dir))
    todo = [(s, d) for s, d in ckpts if all(r["step"] != s for r in done["rows"])]
    print(f"checkpoints: {[s for s, _ in ckpts]}; to score: {[s for s, _ in todo]}", flush=True)
    if not todo:
        return 0

    dev = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(BASE, revision=REVISION, torch_dtype=torch.float16,
                                                   safety_checker=None, requires_safety_checker=False).to(dev)
    sched = DDPMScheduler.from_pretrained(BASE, revision=REVISION, subfolder="scheduler")
    pipe.set_progress_bar_config(disable=True)

    # fixed latents, text embeddings, timesteps and noise for every (split, image, draw)
    g = torch.Generator(device="cpu").manual_seed(42)
    cache = {}
    with torch.no_grad():
        for split, items in sets.items():
            lat, emb = [], []
            for path, text in items:
                im = Image.open(path).convert("RGB").resize((512, 512), Image.BICUBIC)
                x = torch.from_numpy(np.asarray(im, np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None].to(dev, torch.float16)
                lat.append(pipe.vae.encode(x).latent_dist.sample(generator=None) * pipe.vae.config.scaling_factor)
                ids = pipe.tokenizer(text, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt").input_ids.to(dev)
                emb.append(pipe.text_encoder(ids)[0])
            lat = torch.cat(lat).repeat_interleave(N_DRAWS, 0)
            emb = torch.cat(emb).repeat_interleave(N_DRAWS, 0)
            t = torch.randint(0, sched.config.num_train_timesteps, (len(lat),), generator=g)
            noise = torch.randn(lat.shape, generator=g).to(dev, torch.float16)
            noisy = sched.add_noise(lat.float(), noise.float(), t.to(dev)).half()
            cache[split] = (noisy, emb, t.to(dev), noise)

    for step, d in todo:
        t0 = time.time()
        if d is not None:
            pipe.load_lora_weights(str(d), weight_name="pytorch_lora_weights.safetensors")
        row = {"step": step, "source": str(d.name if d is not None else "base")}
        with torch.no_grad():
            for split, (noisy, emb, t, noise) in cache.items():
                per = []
                for b in range(0, len(noisy), BATCH):
                    pred = pipe.unet(noisy[b:b + BATCH], t[b:b + BATCH], encoder_hidden_states=emb[b:b + BATCH]).sample
                    per.append(((pred.float() - noise[b:b + BATCH].float()) ** 2).mean(dim=(1, 2, 3)).cpu())
                per = torch.cat(per).numpy()
                row[f"{split}_loss"] = float(per.mean())
                row[f"{split}_loss_se"] = float(per.std(ddof=1) / np.sqrt(len(per)))
        if d is not None:
            pipe.unload_lora_weights()
        done["rows"].append(row)
        done["rows"].sort(key=lambda r: r["step"])
        done.update({"run_id": RUN_ID, "n": {k: len(v) for k, v in sets.items()}, "n_draws": N_DRAWS})
        out_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(f"step {step}: train {row['train_loss']:.5f}  val {row['val_loss']:.5f}  ({time.time() - t0:.0f} s)",
              flush=True)
    print("=== lora loss done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
