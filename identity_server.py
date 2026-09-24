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

    {"op": "generate", "crop_path": "<path>", "seed": <int>}
    -> {"result": "<path to a generated identity image>"}
    -> {"result": null}   # no face found in this particular crop —
                          # legitimate, caller retries on a later frame
    -> {"error": "..."}

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
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "vendor" / "blanket-infant-face-anonym"))

from rpc_server import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--device", default=None, help="cuda/mps/cpu — default: auto-detect")
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

    output_dir = _HERE / "output" / "identities"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[identity_server] loading StableDiffusionAnonymizer (SDXL+ControlNet+refiner)...", flush=True)
    anonymizer = StableDiffusionAnonymizer(device=args.device)
    print("[identity_server] ready", flush=True)

    def generate(crop_path: str, seed: int):
        image = cv2.imread(crop_path)
        if image is None:
            raise RuntimeError(f"could not read {crop_path}")

        face_detector = DetectorFactory.create_face_detector(FaceDetectorModule.YOLO)
        face_detections = face_detector.detect(image)
        if len(face_detections) == 0:
            return None  # legitimate "no face" — this project's own caller
                         # falls back to passthrough and retries next frame
        face_bbox = face_detections[0].left_top_right_bottom

        landmarks_detector = DetectorFactory.create_facial_landmarks_detector(FacialLandmarksDetectorModule.SPIGA)
        landmarks_detection = landmarks_detector.detect(image, face_detections[0])
        face_landmarks = landmarks_detection.landmarks

        del face_detector, landmarks_detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        orig_h, orig_w = image.shape[:2]

        # Real upstream bug workaround — see module docstring.
        anonymizer.seed = seed

        mask_path = output_dir / f"identity_{seed}_mask.png"
        synthetic_image = anonymizer.generate(
            image=image, face_bbox=face_bbox, face_landmarks=face_landmarks,
            output_size=(orig_w, orig_h), save_mask_path=str(mask_path),
        )

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
                blended = cv2.seamlessClone(synthetic_bgr, image, mask, center, blend_flag)
                synthetic_image = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

        identity_path = output_dir / f"identity_{seed}.jpg"
        synthetic_image.save(identity_path)
        return str(identity_path)

    serve(args.socket, {"generate": generate})


if __name__ == "__main__":
    main()
