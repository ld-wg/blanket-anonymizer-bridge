#!/usr/bin/env python3
"""IdentityGenerator server — wraps BLANKET's own real
`blanket.anonymization.pipelines.image_pipeline` internals as a long-lived
process, so Stable Diffusion XL + ControlNet + refiner load exactly once,
not once per (frame, track) call.

This is the ONLY place in this repository that imports BLANKET's own
GPL-3.0 source directly — see this repo's own README/NOTICE.md for why
that's confined to this repo and never crosses into
`lose-the-faces-keep-the-lesson`'s own process.

Protocol (see `rpc_server.py`). One op:

    {"op": "generate", "crop_path": "<path>", "seed": <int>,
     "box": [l, t, r, b] | null, "tag": "<str>" | null}
    -> {"result": "<path to a generated identity image>"}
    -> {"result": null}   # no face found in this particular crop —
                          # legitimate, caller retries on a later frame
    -> {"error": "..."}

`box` (optional, crop coordinates): the calling project's own Phase 1
detection for this crop. When given, BLANKET's YOLO face detector is
skipped and the box goes straight to SPIGA as a `FaceDetection` — BLANKET's
own detector found no face in any crop of 11 of the 24 tracks on
`video-demo-2.mov` although Phase 1 did (see NOTICE.md). `tag` (optional)
is appended to the output file name, so several attempts for one seed
(different candidate crops) do not overwrite each other.

**Why this doesn't just call BLANKET's own
`image_pipeline.generate_synthetic_identity()` directly**: that function
constructs a brand-new `StableDiffusionAnonymizer` (reloading the whole
SDXL+ControlNet+refiner pipeline) on every single call, and calls
`anonymizer.unload()` at the end of every call — exactly the per-call
reload cost a persistent server exists to avoid. This module instead
constructs ONE `StableDiffusionAnonymizer` at startup (kept warm across
every RPC) and re-implements `generate_synthetic_identity()`'s own
surrounding logic (face/landmark detection, the seed workaround below, and
the same Poisson-blend post-step, gated on the same config keys BLANKET's
own YAML already sets) around that single warm instance. The actual
diffusion inference itself (`anonymizer.generate(...)`) is still BLANKET's
own real, unmodified method — nothing about the model or its weights is
reimplemented here.

**Real upstream bug worked around, not patched**: BLANKET's own
`stable_diffusion_parameters.yaml` hardcodes `seed: 1`, and
`StableDiffusionAnonymizer.generate()` takes no per-call seed argument.
Setting `anonymizer.seed` as a public attribute before calling `.generate()`
is a legitimate external use of a public API surface, not a source
modification — see `lose-the-faces-keep-the-lesson`'s
`models/blanket/NOTICE.md` for the full writeup.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BLANKET_ROOT = _HERE / "vendor" / "blanket-infant-face-anonym"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BLANKET_ROOT))

# BLANKET's own DetectorFactory (blanket/core/detectors/detector_factory.py)
# resolves its config files via BARE relative Path objects
# (`Path("blanket/configs/detector_parameters/...")`, not derived from
# `__file__`) — confirmed via a real FileNotFoundError when this server
# was run with its own repo (blanket-anonymizer-bridge) as the working
# directory, not BLANKET's own. Their own `run_video.py`/`run_image_anonymization.py`
# implicitly assume being invoked with BLANKET's own repo root as CWD.
# Matching that assumption here — not a source patch, just running from the
# directory their own code expects.
os.chdir(_BLANKET_ROOT)

from rpc_server import serve  # noqa: E402


# BLANKET's own real prompt (stable_diffusion_parameters.yaml) is
# literally "high quality photo of a BABY face..." — this project's own
# real-video calibration (2026-09-24) found the generated identities skew
# strongly infant-like regardless of the actual subject's age, consistent
# with that explicit wording (the checkpoint itself,
# diffusers/stable-diffusion-xl-1.0-inpainting-0.1, is a generic public
# SDXL inpainting release — NOT trained/fine-tuned on an infant dataset;
# the bias traces to this one prompt phrase, not the model weights). This
# project's own subjects are not infants, so the default here drops "baby"
# for an age-neutral phrase, keeping every other quality/style descriptor
# from BLANKET's own real prompt unchanged. Overridable via --prompt for
# direct A/B comparison against the original wording.
_DEFAULT_PROMPT = (
    "high quality photo of a person's face, realistic, natural lighting, "
    "clear features, natural skin tone, soft features, no makeup"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default=None, help="cuda/mps/cpu — default: auto-detect")
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT,
                        help="Overrides BLANKET's own real prompt (which says 'baby face' "
                             "explicitly) — pass BLANKET's original wording here to A/B against it")
    parser.add_argument("--negative-prompt", default=None,
                        help="default: BLANKET's own real negative_prompt, unchanged")
    args = parser.parse_args()

    # Everything heavy is imported lazily, inside main(), so `--help` and
    # basic argument errors don't pay the torch/diffusers import cost.
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from blanket.anonymization.methods.stable_diffusion import StableDiffusionAnonymizer
    from blanket.constants.enums.detection_enums import FaceDetectorModule, FacialLandmarksDetectorModule
    from blanket.core.detectors.detector_factory import DetectorFactory
    from blanket.core.objects.detections import FaceDetection

    # Real diffusers-version incompatibility, confirmed via crashes on real
    # generate() calls (2026-09-24): both `StableDiffusionAnonymizer.
    # _load_pipeline()` AND `SDRefiner.load()` unconditionally call
    # `<pipeline>.enable_vae_slicing()`, but the installed diffusers
    # version doesn't define that convenience method on EITHER
    # `StableDiffusionXLControlNetInpaintPipeline` (the base/inpaint stage,
    # used because our real config always sets `use_controlnet: true`) or
    # `StableDiffusionXLImg2ImgPipeline` (the refiner stage) —
    # `enable_sequential_cpu_offload()`/`enable_attention_slicing(1)`,
    # called just before it in both places, both work fine on both
    # classes. Patched externally on each CLASS, not BLANKET's source
    # files: delegates to the VAE's own real `.enable_slicing()` (what the
    # missing convenience method would have called anyway), preserving the
    # actual memory-saving behavior rather than silently no-op'ing it away
    # — matters here since serra1's GPUs are shared and already under real
    # memory pressure. Covers `StableDiffusionXLInpaintPipeline` too
    # (the no-ControlNet branch, unused by our real config but cheap to
    # cover defensively) since this is plausibly a version-wide gap across
    # every SDXL pipeline class, not specific to one.
    from diffusers import (
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    )
    for _pipeline_cls in (
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    ):
        if not hasattr(_pipeline_cls, "enable_vae_slicing"):
            _pipeline_cls.enable_vae_slicing = lambda self: self.vae.enable_slicing()

    output_dir = _HERE / "output" / "identities"
    output_dir.mkdir(parents=True, exist_ok=True)

    # StableDiffusionAnonymizer.__init__'s own default config_path (used
    # when config_path=None) is off by one directory level — verified
    # empirically (2026-09-24): it resolves to
    # blanket/anonymization/configs/module_parameters/... which doesn't
    # exist; the real file lives at
    # blanket/configs/module_parameters/stable_diffusion_parameters.yaml.
    # BLANKET's own generate_synthetic_identity() never hits this bug
    # because it always computes and passes the correct path itself
    # (3 parents up from image_pipeline.py, not stable_diffusion.py's own
    # 2) — this server does the same thing explicitly, since it
    # constructs StableDiffusionAnonymizer directly rather than going
    # through that wrapper (see module docstring for why).
    config_path = _BLANKET_ROOT / "blanket" / "configs" / "module_parameters" / "stable_diffusion_parameters.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"expected BLANKET's own config at {config_path}")

    print("[identity_server] loading StableDiffusionAnonymizer (SDXL+ControlNet+refiner)...", flush=True)
    anonymizer = StableDiffusionAnonymizer(config_path=str(config_path), device=args.device)
    # Override the prompt/negative_prompt attributes BLANKET's own
    # __init__ already set from its config — same pattern as the `seed`
    # workaround above (a public-attribute override, not a source patch).
    # See _DEFAULT_PROMPT's own comment for why the default here isn't
    # BLANKET's real prompt verbatim.
    anonymizer.prompt = args.prompt
    if args.negative_prompt is not None:
        anonymizer.negative_prompt = args.negative_prompt
    print(f"[identity_server] prompt: {anonymizer.prompt!r}", flush=True)
    print("[identity_server] ready", flush=True)

    def generate(crop_path: str, seed: int, box=None, tag=None):
        image = cv2.imread(crop_path)
        if image is None:
            raise RuntimeError(f"could not read {crop_path}")

        if box is not None:
            # Caller's own detection (see module docstring). BLANKET's own
            # FaceDetection object, built directly — its constructor only
            # rounds the box to int32; no source change involved.
            face_detection = FaceDetection(np.asarray(box, dtype=np.float64), confidence=1.0)
        else:
            face_detector = DetectorFactory.create_face_detector(FaceDetectorModule.YOLO)
            face_detections = face_detector.detect(image)
            del face_detector
            if len(face_detections) == 0:
                return None  # legitimate "no face" — this project's own caller
                             # falls back to passthrough and retries next frame
            face_detection = face_detections[0]
        face_bbox = face_detection.left_top_right_bottom

        landmarks_detector = DetectorFactory.create_facial_landmarks_detector(FacialLandmarksDetectorModule.SPIGA)
        landmarks_detection = landmarks_detector.detect(image, face_detection)
        face_landmarks = landmarks_detection.landmarks

        del landmarks_detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        orig_h, orig_w = image.shape[:2]

        # Real upstream bug workaround — see module docstring.
        anonymizer.seed = seed

        # Real upstream bug: generate_synthetic_identity()/generate() both
        # pass output_size=(orig_w, orig_h) — the CROP's own raw
        # dimensions — straight through to the refiner stage. Confirmed
        # crashing (2026-09-24) with `ValueError: operands could not be
        # broadcast together` inside SDRefiner.refine()'s own compositing
        # line: the SDXL img2img refiner pipeline doesn't reliably
        # preserve odd/small input resolutions (our crops are far smaller
        # and less "round" than BLANKET's own square demo images, which
        # never triggered this), so its own output ends up a few pixels
        # off from the mask saved before the refiner ran. Worked around by
        # generating at BLANKET's own native, config-declared resolution
        # (896x896 by default — safely divisible, what the refiner is
        # actually tuned for) instead of the crop's own odd dimensions,
        # then resizing the result back down to the crop's size ourselves
        # before compositing — not a source patch, just not feeding their
        # code a resolution it doesn't handle correctly.
        native_w = anonymizer.config.get("width", 896)
        native_h = anonymizer.config.get("height", 896)
        stem = f"identity_{seed}_{tag}" if tag else f"identity_{seed}"
        mask_path = output_dir / f"{stem}_mask.png"
        synthetic_image = anonymizer.generate(
            image=image, face_bbox=face_bbox, face_landmarks=face_landmarks,
            output_size=(native_w, native_h), save_mask_path=str(mask_path),
        )
        synthetic_image = synthetic_image.resize((orig_w, orig_h), Image.LANCZOS)
        if mask_path.is_file():
            mask_native = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(mask_path), cv2.resize(mask_native, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR))

        # Same Poisson color-blend post-step generate_synthetic_identity()
        # itself does, gated on the same config keys — reused here, not
        # reinvented, since we bypass that wrapper function to stay warm.
        use_poisson = anonymizer.config.get("use_poisson_blending", False)
        poisson_mode = anonymizer.config.get("poisson_blend_mode", "NORMAL")
        if use_poisson and mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            synthetic_bgr = cv2.cvtColor(np.array(synthetic_image), cv2.COLOR_RGB2BGR)
            moments = cv2.moments(mask)
            if moments["m00"] != 0:
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                blend_flag = cv2.NORMAL_CLONE if poisson_mode == "NORMAL" else cv2.MIXED_CLONE
                # Real crash found on video-demo-2.mov (2026-09-24):
                # cv2.seamlessClone requires the mask's own bounding region,
                # placed at `center`, to stay fully inside `image`'s bounds
                # -- a face close to the crop's own edge (itself close to
                # the source video's frame edge, since crop_box() clamps
                # there) can produce a mask whose extent pokes past it,
                # raising a hard cv2.error. Neither this project's own
                # generate_synthetic_identity() port nor, as far as we
                # verified, BLANKET's own real function guards against
                # this -- it's a latent bug in the upstream mechanism
                # itself, just never triggered by their own centered square
                # demo images. Falls back to a plain hard-mask paste,
                # mirroring the exact fallback models/_compositing.py's
                # own poisson_composite() already uses for the same
                # real-world failure mode -- not a new pattern.
                try:
                    blended = cv2.seamlessClone(synthetic_bgr, image, mask, center, blend_flag)
                    synthetic_image = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
                except cv2.error as e:
                    print(f"[identity_server] seamlessClone failed ({e}) -- "
                          "falling back to a hard-mask paste for this identity", flush=True)
                    mask_bool = mask.astype(bool)
                    fallback = image.copy()
                    fallback[mask_bool] = synthetic_bgr[mask_bool]
                    synthetic_image = Image.fromarray(cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB))

        identity_path = output_dir / f"{stem}.jpg"
        synthetic_image.save(identity_path)
        return str(identity_path)

    serve(args.socket, {"generate": generate})


if __name__ == "__main__":
    main()
