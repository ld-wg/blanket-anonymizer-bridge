"""P1 — ArcFace guidance inside BLANKET's SDXL identity generation (the calling
project's contribution plan, D8 / Step 6).

MPGD-style (arXiv:2311.16424): at selected denoising steps, the predicted
clean latent is pushed away from the real person's identity, with no
backpropagation through the UNet:

    x0   = x_t − σ_t · ε                       (Euler, epsilon prediction)
    loss = relu(cos(ArcFace(align(VAE.decode(x0_face))), e_real) − τ)
    x0' = x0 − c · ĝ                            (ĝ: the gradient, scaled to unit RMS, face region only)
    ε'  = (x_t − x0') / σ_t  →  the scheduler's own step(ε', t, x_t)

**How it is installed.**
- It wraps `pipeline.scheduler.step`, an instance attribute, so BLANKET's
  `generate()` and diffusers' loop run unchanged.
- With no guidance state set, the wrapper calls the original step, so the
  control arm runs the identical scheduler.
- The refiner has its own scheduler and is not guided.

**Models.**
- The face is decoded by a separate fp32 copy of BLANKET's VAE
  (`madebyollin/sdxl-vae-fp16-fix`, already in the HF cache), kept on the
  GPU, outside the pipeline's sequential CPU offload.
- ArcFace is the calling project's `w600k_r50.onnx`, converted with
  onnx2torch (Apache-2.0), so the push lives in the same embedding space as
  the track identity and FaceFusion's recognizer.

`e_real` is biometric data. It arrives over the RPC socket, stays in memory
for one call, and is never logged or written. Only scalar cosines are
printed.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Optional

import numpy as np

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
ARCFACE_SIZE = 112
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 similarity mapping src -> dst (least squares), as insightface's estimate_norm."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ e @ vt
    scale = np.trace(np.diag(sig) @ e) / (s ** 2).sum(1).mean()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]])


def affine_sample(image, out_to_in: np.ndarray, out_size: int):
    """Differentiable warp: `out_to_in` maps output pixels to input pixels (pixel centers)."""
    import torch
    import torch.nn.functional as F

    _, _, h, w = image.shape
    a = np.vstack([np.asarray(out_to_in, np.float64), [0, 0, 1]])
    to_pixel_out = np.array([[out_size / 2, 0, out_size / 2 - 0.5], [0, out_size / 2, out_size / 2 - 0.5], [0, 0, 1]])
    to_norm_in = np.array([[2 / w, 0, 1 / w - 1], [0, 2 / h, 1 / h - 1], [0, 0, 1]])
    theta = torch.as_tensor((to_norm_in @ a @ to_pixel_out)[:2], dtype=image.dtype, device=image.device)[None]
    grid = F.affine_grid(theta, (1, image.shape[1], out_size, out_size), align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)


class FaceGuidance:
    def __init__(self, pipeline, device: str = "cuda", arcface_onnx: Optional[Path] = None,
                 gradient_checkpointing: bool = True):
        import onnx2torch
        import torch
        from diffusers import AutoencoderKL

        self.device = device
        self.vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float32).to(device)
        self.vae.requires_grad_(False)
        if gradient_checkpointing:
            self.vae.enable_gradient_checkpointing()
        path = arcface_onnx or Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
        self.arcface = onnx2torch.convert(str(path)).eval().requires_grad_(False).to(device)
        self.scheduler = pipeline.scheduler
        self.state: Optional[dict] = None
        original = self.scheduler.step

        @functools.wraps(original)
        def step(model_output, timestep, sample, *args, **kwargs):
            return self._step(original, model_output, timestep, sample, *args, **kwargs)

        self.scheduler.step = step

    def set(self, e_real, landmarks: np.ndarray, image_size: tuple[int, int], scale: float, tau: float,
            window: int) -> None:
        """Guide the next generation. `landmarks`: the 5 points in the generation canvas
        (image_size, e.g. 896x896); `window`: guide the last N steps (0 = all)."""
        import torch

        e = torch.as_tensor(np.asarray(e_real, np.float32), device=self.device)
        w, h = image_size
        lm = np.asarray(landmarks, np.float64)
        # latent face region: 3.2x the 5 points' spread (eyes to mouth ~ 0.4 of a face), >= 8 cells
        cx, cy = lm.mean(0) / 8
        half = max(4.0, 1.6 * max(np.ptp(lm[:, 0]), np.ptp(lm[:, 1])) / 8)
        x0, y0 = int(max(0, math.floor(cx - half))), int(max(0, math.floor(cy - half)))
        x1, y1 = int(min(w // 8, math.ceil(cx + half))), int(min(h // 8, math.ceil(cy + half)))
        s_inv = np.vstack([cvt_invert(umeyama(lm, ARCFACE_DST)), [0, 0, 1]])
        shift = np.array([[1, 0, -8 * x0], [0, 1, -8 * y0], [0, 0, 1]], np.float64)
        self.state = {"e_real": e / e.norm(), "region": (x0, y0, x1, y1), "out_to_in": (shift @ s_inv)[:2],
                      "scale": scale, "tau": tau, "window": window, "calls": 0, "cos": []}

    def clear(self) -> list[float]:
        cos = self.state["cos"] if self.state else []
        self.state = None
        return cos

    def _step(self, original, model_output, timestep, sample, *args, **kwargs):
        import torch

        st = self.state
        if st is None:
            return original(model_output, timestep, sample, *args, **kwargs)
        sched = self.scheduler
        if sched.step_index is None:
            sched._init_step_index(timestep)
        begin = sched.begin_index or 0
        n_loop = len(sched.timesteps) - begin
        i = st["calls"]
        st["calls"] += 1
        if st["window"] and i < n_loop - st["window"]:
            return original(model_output, timestep, sample, *args, **kwargs)

        sigma = sched.sigmas[sched.step_index].to(sample.device, torch.float32)
        x_t = sample.float()
        x0 = x_t - sigma * model_output.float()
        lx0, ly0, lx1, ly1 = st["region"]
        with torch.enable_grad():
            z = x0[:, :, ly0:ly1, lx0:lx1].detach().clone().requires_grad_(True)
            img = self.vae.decode(z / self.vae.config.scaling_factor).sample
            aligned = affine_sample((img + 1) / 2, st["out_to_in"], ARCFACE_SIZE)
            emb = torch.nn.functional.normalize(self.arcface(2 * aligned - 1), dim=1)[0]
            cos = emb @ st["e_real"]
            loss = torch.relu(cos - st["tau"])
            grad = torch.autograd.grad(loss, z)[0] if float(loss) > 0 else None
        st["cos"].append(round(float(cos), 4))
        if grad is not None:
            g = grad / (grad.norm() + 1e-8) * math.sqrt(grad.numel())   # unit RMS in latent units
            x0[:, :, ly0:ly1, lx0:lx1] -= st["scale"] * g
            model_output = ((x_t - x0) / sigma).to(model_output.dtype)
        return original(model_output, timestep, sample, *args, **kwargs)


def cvt_invert(m: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine."""
    a = np.vstack([m, [0, 0, 1]])
    return np.linalg.inv(a)[:2]
