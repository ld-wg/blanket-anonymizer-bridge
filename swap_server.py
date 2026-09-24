#!/usr/bin/env python3
"""FaceSwapper server — wraps BLANKET's own real, vendored
`blanket.anonymization.methods.facefusion.FaceFusionDirectAnonymizer` as a
long-lived process, so FaceFusion's own model loading (face detector,
landmarker, recognizer, swapper, enhancer) happens once per identity, not
once per frame.

This is the ONLY other place in this repository that imports BLANKET's own
GPL-3.0 source directly (see `identity_server.py`'s docstring for the same
rationale).

Protocol (see `rpc_server.py`). One op:

    {"op": "swap", "identity_path": "<path>", "crop_path": "<path>"}
    -> {"result": "<path to the swapped crop>"}
    -> {"result": null}   # FaceFusion's own internal yolo_face redetection
                          # found no face in this particular crop —
                          # legitimate, caller falls back to passthrough
    -> {"error": "..."}

**Config: BLANKET's own real, unmodified
`blanket/configs/module_parameters/facefusion_parameters.yaml` is used
as-is — no override authored here.** Verified directly (2026-09-24) that
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
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BLANKET_ROOT = _HERE / "vendor" / "blanket-infant-face-anonym"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BLANKET_ROOT))
sys.path.insert(0, str(_BLANKET_ROOT / "external" / "facefusion"))

from rpc_server import serve  # noqa: E402

_CONFIG_PATH = _BLANKET_ROOT / "blanket" / "configs" / "module_parameters" / "facefusion_parameters.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()

    import cv2

    from blanket.anonymization.methods.facefusion import FaceFusionDirectAnonymizer

    output_dir = _HERE / "output" / "swapped"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"BLANKET's own facefusion_parameters.yaml not found at {_CONFIG_PATH} — "
            "check the vendor/blanket-infant-face-anonym submodule is initialized."
        )

    # One FaceFusionDirectAnonymizer instance per identity_path, cached —
    # its own __init__ does real, heavy one-time setup (pre_check() for
    # every enabled processor, computing the source face's ArcFace
    # embedding) keyed to one synthetic_face_path. A new identity means a
    # new instance; the same identity across every frame of one track
    # reuses one, which is also what makes iou_filter's own frame-to-frame
    # continuity state meaningful (see module docstring).
    _anonymizers: dict[str, "FaceFusionDirectAnonymizer"] = {}
    _counter = {"n": 0}

    def _get_anonymizer(identity_path: str) -> "FaceFusionDirectAnonymizer":
        anonymizer = _anonymizers.get(identity_path)
        if anonymizer is None:
            print(f"[swap_server] loading FaceFusionDirectAnonymizer for {identity_path}...", flush=True)
            anonymizer = FaceFusionDirectAnonymizer(
                synthetic_face_path=identity_path, config_path=str(_CONFIG_PATH),
            )
            _anonymizers[identity_path] = anonymizer
        return anonymizer

    def swap(identity_path: str, crop_path: str):
        anonymizer = _get_anonymizer(identity_path)
        image = cv2.imread(crop_path)
        if image is None:
            raise RuntimeError(f"could not read {crop_path}")
        try:
            result_frame, _bboxes = anonymizer.anonymize(image, detections=[])
        except RuntimeError as e:
            if "No faces detected" in str(e):
                return None  # legitimate — caller falls back to passthrough
            raise  # anything else (e.g. "FaceFusion returned unchanged
                   # image") is a real failure, surfaced as {"error": ...}
                   # by rpc_server.py, not silently swallowed

        _counter["n"] += 1
        out_path = output_dir / f"swapped_{_counter['n']:08d}.png"
        cv2.imwrite(str(out_path), result_frame)
        return str(out_path)

    serve(args.socket, {"swap": swap})


if __name__ == "__main__":
    main()
