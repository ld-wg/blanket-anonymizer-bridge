#!/usr/bin/env python3
"""P1 go/no-go: time and peak GPU memory of one guided step's extra work,
in the identity venv, before any P1 experiment.

    decode a face-sized latent crop (fp32 VAE) -> align -> ArcFace -> backward

Run for 48, 64 and 80 latent cells (384–640 px), with and without the VAE's
gradient checkpointing. Go: at most ~3 s per guided step and ~3 GB extra
(the calling project's results plan, 2026-10-01). Prints JSON.

    CUDA_VISIBLE_DEVICES=1 .venv-identity/bin/python tools/p1_spike.py --image output/identities/<any>.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import numpy as np
    import onnx2torch
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image

    from p1_guidance import ARCFACE_SIZE, VAE_ID, affine_sample

    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[48, 64, 80])
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    dev = "cuda"
    arcface = onnx2torch.convert(str(Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx")).eval().requires_grad_(False).to(dev)
    results = []
    for ckpt in (False, True):
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float32).to(dev).requires_grad_(False)
        if ckpt:
            vae.enable_gradient_checkpointing()
        for size in args.sizes:
            img = Image.open(args.image).convert("RGB").resize((8 * size, 8 * size))
            x = torch.from_numpy(np.asarray(img, np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                z0 = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            e = torch.nn.functional.normalize(torch.randn(512, device=dev), dim=0)
            s = 8 * size / ARCFACE_SIZE   # aligned 112 -> the whole decoded crop (a centered face)
            out_to_in = np.array([[s, 0, 0], [0, s, 0]], np.float64)
            times = []
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.repeats):
                t = time.perf_counter()
                z = z0.clone().requires_grad_(True)
                dec = vae.decode(z / vae.config.scaling_factor).sample
                emb = torch.nn.functional.normalize(arcface(2 * affine_sample((dec + 1) / 2, out_to_in, ARCFACE_SIZE) - 1), dim=1)[0]
                g = torch.autograd.grad(torch.relu(emb @ e + 1), z)[0]
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
            results.append({"latent": size, "pixels": 8 * size, "checkpointing": ckpt,
                            "seconds": round(min(times), 3), "peak_extra_gib": round((torch.cuda.max_memory_allocated() - base) / 2**30, 2),
                            "grad_finite": bool(torch.isfinite(g).all())})
        del vae
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
