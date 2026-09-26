#!/usr/bin/env python3
"""FaceSwapper server — wraps BLANKET's own real, vendored
`blanket.anonymization.methods.facefusion.FaceFusionDirectAnonymizer` as a
long-lived process, so FaceFusion's own model loading (face detector,
landmarker, recognizer, swapper, enhancer) happens once per identity, not
once per frame.

This is the ONLY other place in this repository that imports BLANKET's own
GPL-3.0 source directly (see `identity_server.py`'s docstring for the same
rationale).

Protocol (see `rpc_server.py`). Three ops:

    {"op": "swap", "identity_path": "<path>", "crop_path": "<path>"}
    -> {"result": "<path to the swapped crop>" | null}
          # the original op, unchanged, for callers predating the two below

    {"op": "swap_with_reason", "identity_path": "<path>", "crop_path": "<path>"}
    -> {"result": {"path": "<path to the swapped crop>", "reason": null}}
    -> {"result": {"path": null, "reason": "<why>"}}
          # legitimate "nothing to swap", one of:
          #   identity_unusable  — the SDXL-generated identity image itself
          #                        has no face FaceFusion can detect
          #                        (confirmed 2026-09-24: e.g. ControlNet
          #                        faithfully reproducing an extreme head
          #                        angle); permanent for that identity
          #   swap_no_face       — FaceFusion's own yolo_face found no face
          #                        in this target crop; transient
          #   swap_iou_rejected  — BLANKET's iou_filter rejected every face
          #                        in this crop; transient
    -> {"error": "..."}

    {"op": "check_identity", "identity_path": "<path>"}
    -> {"result": true | false}   # does FaceFusion find a face in it?

`check_identity` lets the caller try another candidate crop right after
generating an unusable identity, instead of discovering it at the first
swap and passing the whole track through.

**Config: BLANKET's own real, unmodified
`blanket/configs/module_parameters/facefusion_parameters.yaml` is used
as-is by default.** `--face-detector-score X` is the one override: it
writes a copy of that YAML with only `face_detector_score` changed to
`output/facefusion_parameters.override.yaml` and uses the copy (for the
calling project's detector-threshold sweep; BLANKET ships 0.5). Verified directly (2026-09-24) that
its real shipped defaults already do what this integration needs:
`max_faces: 1` caps output to a single swapped face per crop even though
`face_selector_mode` is hardcoded `'many'` inside
`FaceFusionDirectAnonymizer.__init__` (not config-driven), `iou_filter:
true` is a meaningful continuity check across one track's own crops over
time (this server caches one `FaceFusionDirectAnonymizer` instance per
`identity_path`, i.e. per track), and `execution_providers: [coreml, cuda,
cpu]` already prioritizes GPU correctly.

**`detections` is intentionally never populated** — verified directly
against `FaceFusionDirectAnonymizer.anonymize()`'s real source that this
parameter is dead code, never read in the method body. Per-frame face
redetection always happens internally via FaceFusion's own `yolo_face`;
substituting this project's own Phase 1 detections there would require
patching BLANKET's source, which this integration deliberately does not do.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BLANKET_ROOT = _HERE / "vendor" / "blanket-infant-face-anonym"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BLANKET_ROOT))
sys.path.insert(0, str(_BLANKET_ROOT / "external" / "facefusion"))

# Precautionary, mirroring identity_server.py's confirmed fix: BLANKET's own
# DetectorFactory relies on bare relative paths assuming its own repo root
# as CWD (confirmed there via a real FileNotFoundError). Not yet
# independently confirmed whether FaceFusion's own code has the same
# assumption — cheap to match anyway, and this is the directory their own
# `run_video.py` expects to run from.
os.chdir(_BLANKET_ROOT)

from rpc_server import serve  # noqa: E402

_CONFIG_PATH = _BLANKET_ROOT / "blanket" / "configs" / "module_parameters" / "facefusion_parameters.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--face-detector-score", type=float, default=None,
                        help="override BLANKET's face_detector_score (0.5) in a copy of its YAML")
    args = parser.parse_args()

    import cv2
    import yaml

    from blanket.anonymization.methods.facefusion import FaceFusionDirectAnonymizer

    output_dir = _HERE / "output" / "swapped"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"BLANKET's own facefusion_parameters.yaml not found at {_CONFIG_PATH} — "
            "check the vendor/blanket-infant-face-anonym submodule is initialized."
        )
    config_path = _CONFIG_PATH
    if args.face_detector_score is not None:
        config = yaml.safe_load(_CONFIG_PATH.read_text())
        config["face_detector_score"] = args.face_detector_score
        config_path = _HERE / "output" / "facefusion_parameters.override.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"[swap_server] FaceFusion config: {config_path}", flush=True)

    # One FaceFusionDirectAnonymizer instance per identity_path, cached —
    # its own __init__ does real, heavy one-time setup (pre_check() for
    # every enabled processor, computing the source face's ArcFace
    # embedding) keyed to one synthetic_face_path. A new identity means a
    # new instance; the same identity across every frame of one track
    # reuses one, which is also what makes iou_filter's own frame-to-frame
    # continuity state meaningful (see module docstring).
    # Cache holds a real anonymizer, OR `False` for an identity_path
    # confirmed to have no detectable face (see below) — `False`, not
    # missing-from-dict, so a doomed identity's expensive-but-failing
    # FaceFusionDirectAnonymizer construction (pre_check() for every
    # enabled processor already runs BEFORE the "no face" check inside
    # BLANKET's own __init__) is never retried on every subsequent frame
    # of the same track.
    _anonymizers: dict[str, object] = {}
    _counter = {"n": 0}

    def _get_anonymizer(identity_path: str):
        if identity_path in _anonymizers:
            return _anonymizers[identity_path]  # real anonymizer, or False
        print(f"[swap_server] loading FaceFusionDirectAnonymizer for {identity_path}...", flush=True)
        try:
            anonymizer = FaceFusionDirectAnonymizer(
                synthetic_face_path=identity_path, config_path=str(config_path),
            )
        except ValueError as e:
            # Real, confirmed case (2026-09-24): FaceFusionDirectAnonymizer's
            # own __init__ raises this when its face_analyser finds zero
            # faces in the identity image itself — see module docstring.
            if "No face detected in source" in str(e):
                print(f"[swap_server] no usable face in identity image {identity_path} "
                      "-- this track will passthrough for its whole duration", flush=True)
                _anonymizers[identity_path] = False
                return False
            raise
        _anonymizers[identity_path] = anonymizer
        return anonymizer

    def check_identity(identity_path: str) -> bool:
        return _get_anonymizer(identity_path) is not False

    def swap(identity_path: str, crop_path: str):
        return swap_with_reason(identity_path, crop_path)["path"]

    def swap_with_reason(identity_path: str, crop_path: str):
        anonymizer = _get_anonymizer(identity_path)
        if anonymizer is False:
            return {"path": None, "reason": "identity_unusable"}
        image = cv2.imread(crop_path)
        if image is None:
            raise RuntimeError(f"could not read {crop_path}")
        try:
            result_frame, _bboxes = anonymizer.anonymize(image, detections=[])
        except RuntimeError as e:
            # Both confirmed real on video-demo-2.mov (2026-09-24), both
            # legitimate per-frame "nothing to swap here" outcomes in
            # BLANKET's own real anonymize() -- neither is cached (unlike
            # the "no face in identity image" case above): a transient
            # per-frame miss, not a permanent property of this identity,
            # so the very next frame gets a fresh attempt.
            #
            # "No faces detected": FaceFusion's own yolo_face found
            # nothing in this crop.
            #
            # "IoU filter rejected all faces": BLANKET's own iou_filter
            # (real, on by default in facefusion_parameters.yaml) rejects
            # a detected face whose bbox doesn't match this identity's own
            # previous_bboxes closely enough (config: iou_threshold=0.4) --
            # BLANKET's own VideoPipeline.run() handles this by reusing the
            # last successful frame's output; this project instead falls
            # back to passthrough for just this one frame, matching this
            # project's own existing no-usable-face contract rather than
            # adding new per-identity "last successful swap" state to
            # replicate upstream's exact fallback.
            if "No faces detected" in str(e):
                return {"path": None, "reason": "swap_no_face"}
            if "IoU filter rejected all faces" in str(e):
                return {"path": None, "reason": "swap_iou_rejected"}
            raise  # anything else (e.g. "FaceFusion returned unchanged
                   # image") is a real failure, surfaced as {"error": ...}
                   # by rpc_server.py, not silently swallowed

        _counter["n"] += 1
        out_path = output_dir / f"swapped_{_counter['n']:08d}.png"
        cv2.imwrite(str(out_path), result_frame)
        return {"path": str(out_path), "reason": None}

    serve(args.socket, {"swap": swap, "swap_with_reason": swap_with_reason, "check_identity": check_identity})


if __name__ == "__main__":
    main()
