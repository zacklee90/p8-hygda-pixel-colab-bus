"""M4 arm A3-RIS: generate eval images with Φdet applied INSIDE the denoising loop (repair-in-sampling).

Same model, scheduler (the pipeline default), steps, guidance, captions and seeds as ``generate_lora.py``, so each
image is paired 1:1 with its post-hoc-only twin. At the end of the steps listed in RIS_STEPS (default 10,20 of 30):

1. one extra CFG UNet call on the current latents x_t gives eps; x0_hat = (x_t - sqrt(1-a_t) eps) / sqrt(a_t);
2. x0_hat is decoded, colour-snapped to the 16x16 grid and passed through Φdet (``hygdap.detail.detail_grid``:
   chirality mirror, repair, budget projection) with the prompt's Ψphys theta, then re-rendered and re-encoded -> x0';
3. x_t <- sqrt(a_t) x0' + sqrt(1-a_t) eps  (the clean component is replaced, the noise component kept), and
   sampling continues. Φdet is still applied after the last step as usual (offline, ``p3_eval_generation --score``).

Env (on top of the bus's): RUN_ID (LoRA weights at runs/<RUN_ID>), PROMPTS_CSV (id, file, seed, caption,
theta_json), RENDER_JSON (the ``render`` config section), HYGDAP_ZIP (zip of the hygdap package, pure python),
RIS_STEPS, STEPS, GUIDANCE.
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
PROMPTS_CSV = Path(os.environ["PROMPTS_CSV"])
RENDER_JSON = Path(os.environ.get("RENDER_JSON", str(PROMPTS_CSV.parent / "render.json")))
HYGDAP_ZIP = os.environ.get("HYGDAP_ZIP", str(DRIVE_ROOT / "code" / "hygdap.zip"))
LORA_DIR = Path(os.environ.get("LORA_DIR", str(DRIVE_ROOT / "runs" / RUN_ID)))
STEPS = int(os.environ.get("STEPS", "30"))
GUIDANCE = float(os.environ.get("GUIDANCE", "7.5"))
RIS_STEPS = {int(s) for s in os.environ.get("RIS_STEPS", "10,20").split(",") if s.strip()}
LIMIT = int(os.environ.get("LIMIT", "0"))                      # >0: only the first LIMIT rows (smoke tests)
BASE = os.environ.get("MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")
REVISION = os.environ.get("MODEL_REVISION", "451f4fe16113bff5a5d2269ed5ad43b0592e9a14") or None
NEGATIVE = os.environ.get("NEGATIVE_PROMPT",
                          "blurry, low quality, distorted, artistic, painting, watermark, text, numbers, photorealistic")

sys.path.insert(0, str(Path(__file__).resolve().parent))
if os.path.isfile(HYGDAP_ZIP):
    sys.path.insert(0, HYGDAP_ZIP)
from envfix import neutralize_torchao  # noqa: E402


def main() -> int:
    neutralize_torchao()
    import numpy as np
    import torch
    from diffusers import StableDiffusionPipeline

    from hygdap.detail import detail_grid
    from hygdap.physics.cp_pixel import CPPixelDesign
    from hygdap.render.pixel_render import grid_from_image, render_pixel_layout

    rc = json.loads(RENDER_JSON.read_text(encoding="utf-8"))
    weights = LORA_DIR / "pytorch_lora_weights.safetensors"
    for p in (weights, PROMPTS_CSV):
        if not p.is_file():
            print(f"DATA_MISSING: {p} not found", flush=True)
            return 1
    out_dir = PROMPTS_CSV.parent / "gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(PROMPTS_CSV, encoding="utf-8")))
    if LIMIT:
        rows = rows[:LIMIT]
    todo = [r for r in rows if not (out_dir / r["file"]).is_file()]
    print(f"{len(rows)} prompts, {len(todo)} to generate, RIS at steps {sorted(RIS_STEPS)} of {STEPS}", flush=True)
    if not todo:
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(BASE, revision=REVISION, torch_dtype=dtype,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.load_lora_weights(str(LORA_DIR))
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    alphas = pipe.scheduler.alphas_cumprod.to(device)

    def project(latents, t, prompt_embeds, design):
        """Replace the clean component of x_t by its Φdet projection; returns new latents and a report."""
        lat_in = torch.cat([latents] * 2) if GUIDANCE > 1 else latents
        lat_in = pipe.scheduler.scale_model_input(lat_in, t)
        with torch.no_grad():
            eps = pipe.unet(lat_in, t, encoder_hidden_states=prompt_embeds).sample
        if GUIDANCE > 1:
            e_u, e_c = eps.chunk(2)
            eps = e_u + GUIDANCE * (e_c - e_u)
        a = alphas[int(t)].to(latents.dtype)
        x0 = (latents - (1 - a).sqrt() * eps) / a.sqrt()
        with torch.no_grad():
            img = pipe.vae.decode(x0 / pipe.vae.config.scaling_factor).sample
        pil = pipe.image_processor.postprocess(img, output_type="pil")[0]
        n = design.grid_n
        raw, _ = grid_from_image(pil, rc["colors"], n, rc["px_per_pixel"], rc["resolution"])
        final, rep = detail_grid(raw, design)
        rendered = render_pixel_layout(final, (n // 2 - 1, design.probe_col), rc["colors"], rc["resolution"], rc["px_per_pixel"])
        x = pipe.image_processor.preprocess(rendered).to(device=device, dtype=latents.dtype)
        with torch.no_grad():
            x0p = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
        return a.sqrt() * x0p + (1 - a).sqrt() * eps, {"flips": int(rep.flips_to_final), "mirrored": bool(rep.mirrored)}

    t_all = time.time()
    with open(PROMPTS_CSV.parent / "gen_log.jsonl", "a", encoding="utf-8") as log:
        for k, r in enumerate(todo, 1):
            th = json.loads(r["theta_json"])
            design = CPPixelDesign(**{kk: v for kk, v in th.items() if kk in CPPixelDesign.__dataclass_fields__ and kk != "validity"})
            reports = []

            def cb(p, i, t, kw, design=design, reports=reports):
                if (i + 1) in RIS_STEPS and (i + 1) < STEPS:
                    t_next = p.scheduler.timesteps[i + 1]
                    kw["latents"], rep = project(kw["latents"], t_next, kw["prompt_embeds"], design)
                    reports.append({"step": i + 1, **rep})
                return kw

            gen = torch.Generator(device="cpu").manual_seed(int(r["seed"]))
            t0 = time.time()
            img = pipe(r["caption"], negative_prompt=NEGATIVE, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                       generator=gen, height=512, width=512, callback_on_step_end=cb,
                       callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"]).images[0]
            img.save(out_dir / r["file"])
            log.write(json.dumps({"id": r["id"], "file": r["file"], "seconds": round(time.time() - t0, 2), "steps": STEPS,
                                  "ris": reports, "device": device}) + "\n")
            log.flush()
            if k % 10 == 0 or k == len(todo):
                rate = (time.time() - t_all) / k
                print(f"[{k}/{len(todo)}] {rate:.1f} s/img  ETA {rate * (len(todo) - k) / 60:.1f} min", flush=True)
    print("=== generation done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
