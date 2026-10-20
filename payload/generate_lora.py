"""Generate evaluation images from a trained LoRA on the Colab GPU (bus job ``kind: python``).

Inputs (env):
    RUN_ID            run folder under MyDrive/p8-hygda-pixel/runs/ holding pytorch_lora_weights.safetensors
    PROMPTS_CSV       CSV with columns id,prompt_idx,seed,file,caption (written by the local driver
                      into MyDrive/p8-hygda-pixel/eval/<RUN_ID>/prompts.csv)
    STEPS, GUIDANCE   inference settings (default 30 / 7.5)
    LORA_SCALE        default 1.0

Writes PNGs next to the CSV under ``gen/`` and a ``gen_log.jsonl``; skips images that already
exist, so a cancelled job resumes. The pipeline code is a verbatim copy of
``hygda/gen/sd_pipeline.py`` (this repo has no ``hygda`` package on the Colab side).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("P8_DRIVE_BUS", "/content/drive/MyDrive/p8-hygda-pixel/bus")).parent
RUN_ID = os.environ["RUN_ID"]
PROMPTS_CSV = Path(os.environ.get("PROMPTS_CSV", str(DRIVE_ROOT / "eval" / RUN_ID / "prompts.csv")))
STEPS = int(os.environ.get("STEPS", "30"))
GUIDANCE = float(os.environ.get("GUIDANCE", "7.5"))
LORA_SCALE = float(os.environ.get("LORA_SCALE", "1.0"))
BASE = os.environ.get("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
REVISION = os.environ.get("MODEL_REVISION", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14") or None
NEGATIVE = os.environ.get("NEGATIVE_PROMPT",
                          "blurry, low quality, distorted, artistic, painting, watermark, text, numbers, photorealistic")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from envfix import neutralize_torchao  # noqa: E402


def main() -> int:
    neutralize_torchao()
    import torch
    from diffusers import StableDiffusionPipeline

    lora_dir = DRIVE_ROOT / "runs" / RUN_ID
    weights = lora_dir / "pytorch_lora_weights.safetensors"
    if not weights.is_file():
        print(f"DATA_MISSING: {weights} not found", flush=True)
        return 1
    if not PROMPTS_CSV.is_file():
        print(f"DATA_MISSING: {PROMPTS_CSV} not found", flush=True)
        return 1
    out_dir = PROMPTS_CSV.parent / "gen"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(PROMPTS_CSV, encoding="utf-8")))
    todo = [r for r in rows if not (out_dir / r["file"]).is_file()]
    print(f"{len(rows)} prompts, {len(todo)} to generate", flush=True)
    if not todo:
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE, revision=REVISION, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.load_lora_weights(str(lora_dir))
    if LORA_SCALE != 1.0:
        pipe.fuse_lora(lora_scale=LORA_SCALE)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    print(f"pipeline on {device}, LoRA {weights.name}, steps {STEPS}, guidance {GUIDANCE}", flush=True)

    t_all = time.time()
    with open(PROMPTS_CSV.parent / "gen_log.jsonl", "a", encoding="utf-8") as log:
        for k, r in enumerate(todo, 1):
            gen = torch.Generator(device="cpu").manual_seed(int(r["seed"]))
            t0 = time.time()
            img = pipe(r["caption"], negative_prompt=NEGATIVE, num_inference_steps=STEPS,
                       guidance_scale=GUIDANCE, generator=gen, height=512, width=512).images[0]
            img.save(out_dir / r["file"])
            log.write(json.dumps({"id": r["id"], "file": r["file"], "seconds": round(time.time() - t0, 2),
                                  "steps": STEPS, "device": device}) + "\n")
            log.flush()
            if k % 10 == 0 or k == len(todo):
                rate = (time.time() - t_all) / k
                print(f"[{k}/{len(todo)}] {rate:.1f} s/img  ETA {rate * (len(todo) - k) / 60:.1f} min", flush=True)
    print("=== generation done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
