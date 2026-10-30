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

`swap_with_reason` also takes an optional identity-push mode (the calling
project's contribution step P2):

    "mode": "native" | "none" | "track"   (default "native")
    "push": [512 floats] | null            (required for "track")
    "beta": <float>                        (push strength for "track")

- `native`: BLANKET unchanged. It sets `face_swapper_weight = 100`,
  outside FaceFusion's 0–1 range; `balance_source_embedding` clamps it to
  w = −0.35, so the swap is conditioned on 1.35·synthetic − 0.35·(the real
  face in *this* frame) — a fixed single-frame push away (see NOTICE.md).
- `none`: `face_swapper_weight = 0.5` → w = 0: the synthetic embedding
  alone, no push. The control arm.
- `track`: w = 0 as in `none`, and the source face's embedding is replaced
  by `normalize(ê_synth − beta·p)`, where `p` is the caller's track-level
  estimate of the real person's identity (unit vector in the same ArcFace
  w600k_r50 space FaceFusion's recognizer uses). The push happens in raw
  ArcFace space, before inswapper's `emap` projection, so both terms live
  in one space — unlike `native`'s mix (NOTICE.md, open question).
  Implemented through the anonymizer's public `source_face` attribute
  (FaceFusion's `Face` is a namedtuple), no source change.

`push` is biometric data (an estimate of a real person's face identity):
it is used in memory for one call and never logged or written.

`swap_with_reason` also takes the target face from the caller's own
detector instead of FaceFusion's (the calling project's recall work,
2026-09-30):

    "box": [x1, y1, x2, y2] | null         (crop coordinates)
    "landmarks5": [[x, y] x 5] | null      (eyes, nose, mouth corners)

With both, the target face is built by FaceFusion's own
`face_analyser.create_faces()` from that box and those points (its own
2dfan4 landmarker, recognizer and classifier still run), then swapped and
enhanced by the same processors, in the same order, as BLANKET's
`anonymize()`. BLANKET's per-frame `yolo_face` re-detection and its IoU
filter are skipped: the caller's detector already found the face and its
tracker already provides continuity. Measured on the caller's
video-demo-3.mov: `yolo_face` at 0.5 missed 1417 of 4726 faces the
caller's SCRFD had found (bowed heads, small faces), plus 38 IoU
rejections. Without `box`, the original flow runs unchanged.

`--lenient-identity` (server flag): when `yolo_face` at 0.5 finds no face
in an identity image, retry once with FaceFusion's `many` detector
(retinaface + scrfd + yolo_face) at score 0.25, angle 0 only, and take the
highest-scoring face, the same choice as BLANKET's `source_faces[0]`.
Measured: the rejected identities are small (58–269 px). `many` at 0.25
found the subject's face in all 13 identities rejected on
video-demo-3.mov. Only the identity image gets the lenient detector.

`check_identity` lets the caller try another candidate crop right after
generating an unusable identity, instead of discovering it at the first
swap and passing the whole track through.

`identity_box` (optional, on `check_identity` and `swap_with_reason`; added
2026-10-30): the box, in identity-image coordinates, of the face the
identity server inpainted. The identity image is the caller's context crop
(margin 0.8 of the face box on each side), so it can also hold real,
untouched faces of neighbours. BLANKET takes `source_faces[0]`, the
highest-scoring face, which is not guaranteed to be the inpainted one: a
sharper, larger neighbour would become the swap source, i.e. a real
classmate's identity. With `identity_box`, the source is the detected face
that overlaps it most (IoU), and an identity whose faces all miss it
(IoU < 0.3) is unusable. `check_identity` then also reports which face was
chosen, so the caller can count how often it was not `source_faces[0]`.
Without `identity_box`, BLANKET's choice is kept.

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
redetection always happens internally via FaceFusion's own `yolo_face`.
The `box`/`landmarks5` path above therefore does not go through
`anonymize()` at all; it calls the same FaceFusion functions and BLANKET's
own processor settings from this server, so BLANKET's source stays
unpatched.
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
_MIN_SOURCE_IOU = 0.3


def box_iou(a, b) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def pick_source(boxes, identity_box, min_iou: float = _MIN_SOURCE_IOU):
    """Index of the face box that overlaps `identity_box` most, or None if none
    reaches `min_iou`. `boxes` are in detection-score order (index 0 = BLANKET's choice)."""
    if not boxes:
        return None
    ious = [box_iou(b, identity_box) for b in boxes]
    best = max(range(len(ious)), key=ious.__getitem__)
    return best if ious[best] >= min_iou else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--face-detector-score", type=float, default=None,
                        help="override BLANKET's face_detector_score (0.5) in a copy of its YAML")
    parser.add_argument("--lenient-identity", action="store_true",
                        help="retry identity images yolo_face rejects with FaceFusion's 'many' detector at 0.25")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import yaml

    from blanket.anonymization.methods.facefusion import FaceFusionDirectAnonymizer
    from facefusion import face_analyser, face_detector, state_manager
    from facefusion.face_store import clear_static_faces
    from facefusion.processors.modules.face_enhancer import core as face_enhancer
    from facefusion.processors.modules.face_swapper import core as face_swapper

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
        # Built in two steps so that, when __init__ stops at "No face detected
        # in source" (its last statement), the instance keeps everything it
        # already set up and --lenient-identity can supply the source face.
        anonymizer = FaceFusionDirectAnonymizer.__new__(FaceFusionDirectAnonymizer)
        try:
            anonymizer.__init__(synthetic_face_path=identity_path, config_path=str(config_path))
        except ValueError as e:
            # Real, confirmed case (2026-09-24): FaceFusionDirectAnonymizer's
            # own __init__ raises this when its face_analyser finds zero
            # faces in the identity image itself — see module docstring.
            if "No face detected in source" not in str(e):
                raise
            source_face = _lenient_source_face(identity_path) if args.lenient_identity else None
            if source_face is None:
                print(f"[swap_server] no usable face in identity image {identity_path} "
                      "-- this track will passthrough for its whole duration", flush=True)
                _anonymizers[identity_path] = False
                return False
            print(f"[swap_server] identity {identity_path}: face found by the lenient detector", flush=True)
            anonymizer.source_faces = source_face   # every face, score order (see _lenient_source_face)
            anonymizer.source_face = source_face[0]
        _anonymizers[identity_path] = anonymizer
        return anonymizer

    # identity_path -> index of the chosen source face in score order (-1: none overlaps)
    _source_choice: dict[str, int] = {}

    def _select_source(anonymizer, identity_path: str, identity_box) -> bool:
        """Source face = the detected face that overlaps the inpainted box (see module
        docstring). False when none does: the identity is unusable."""
        if identity_box is None:
            return True
        if identity_path not in _source_choice:
            boxes = [f.bounding_box for f in anonymizer.source_faces]
            idx = pick_source(boxes, identity_box)
            _source_choice[identity_path] = -1 if idx is None else idx
            if idx is None or idx != 0:
                print(f"[swap_server] identity {identity_path}: source face "
                      f"{'none overlaps the inpainted box' if idx is None else f'#{idx}, not #0'} "
                      f"({len(boxes)} faces)", flush=True)
            if idx is not None:
                anonymizer.source_face = anonymizer.source_faces[idx]
                _original_faces[identity_path] = anonymizer.source_face
        return _source_choice[identity_path] >= 0

    def _lenient_source_face(identity_path: str):
        """The highest-scoring face FaceFusion's `many` detector finds at 0.25,
        angle 0 only; FaceFusion state is restored afterwards."""
        saved = {k: state_manager.get_item(k)
                 for k in ("face_detector_model", "face_detector_score", "face_detector_angles")}
        try:
            state_manager.init_item("face_detector_model", "many")
            state_manager.init_item("face_detector_score", 0.25)
            state_manager.init_item("face_detector_angles", [0])
            face_detector.pre_check()
            faces = face_analyser.get_many_faces([cv2.imread(identity_path)])
        finally:
            for k, v in saved.items():
                state_manager.init_item(k, v)
            clear_static_faces()
        return list(faces) if faces else None

    def check_identity(identity_path: str, identity_box=None):
        """Without identity_box: whether FaceFusion finds a face (bool, as before).
        With it: {"usable": bool, "source_index": int} (-1 when no face overlaps the box)."""
        anonymizer = _get_anonymizer(identity_path)
        if identity_box is None:
            return anonymizer is not False
        if anonymizer is False:
            return {"usable": False, "source_index": -1}
        usable = _select_source(anonymizer, identity_path, identity_box)
        return {"usable": usable, "source_index": _source_choice[identity_path]}

    # identity_path -> the source face as FaceFusion computed it, before any
    # push replaced its embedding.
    _original_faces: dict[str, object] = {}
    _modes = {"native": 100, "none": 0.5, "track": 0.5}  # mode -> face_swapper_weight

    def _apply_mode(anonymizer, identity_path: str, mode: str, push, beta: float) -> None:
        if mode not in _modes:
            raise ValueError(f"unknown swap mode {mode!r} (known: {sorted(_modes)})")
        original = _original_faces.setdefault(identity_path, anonymizer.source_face)
        # Global FaceFusion state, and every new FaceFusionDirectAnonymizer
        # resets it to 100 in its __init__ — so it is set on every call.
        state_manager.init_item("face_swapper_weight", _modes[mode])
        if mode != "track":
            anonymizer.source_face = original
            return
        if push is None:
            raise ValueError("swap mode 'track' needs a push embedding")
        synth = np.asarray(original.embedding, dtype=np.float32)
        synth = synth / np.linalg.norm(synth)
        p = np.asarray(push, dtype=np.float32)
        p = p / np.linalg.norm(p)
        pushed = synth - float(beta) * p
        pushed = (pushed / np.linalg.norm(pushed)).astype(np.float32)
        anonymizer.source_face = original._replace(embedding=pushed, embedding_norm=pushed)

    def swap(identity_path: str, crop_path: str):
        return swap_with_reason(identity_path, crop_path)["path"]

    def _swap_given_face(anonymizer, image, box, landmarks5):
        """BLANKET's anonymize() for one face the caller already found: the
        target face from FaceFusion's own create_faces(), then the same
        processors in the same order. Score 1.0: the caller's detector
        already accepted this face, and create_faces() only uses the score
        to pass its NMS step."""
        faces = face_analyser.create_faces(image, [np.asarray(box, dtype=np.float32)], [1.0],
                                           [np.asarray(landmarks5, dtype=np.float32)])
        if not faces:
            raise RuntimeError("No faces detected")
        target_face = faces[0]
        result_frame = face_swapper.swap_face(source_face=anonymizer.source_face, target_face=target_face,
                                              temp_vision_frame=image.copy())
        if anonymizer.enable_expression_restorer:
            from facefusion.processors.modules.expression_restorer import core as expression_restorer
            result_frame = expression_restorer.restore_expression(
                target_face=target_face, target_vision_frame=result_frame, temp_vision_frame=result_frame)
        if anonymizer.enable_face_enhancer:
            result_frame = face_enhancer.enhance_face(target_face=target_face, temp_vision_frame=result_frame)
        if np.array_equal(result_frame, image):
            raise RuntimeError("FaceFusion returned unchanged image")
        return result_frame

    def swap_with_reason(identity_path: str, crop_path: str, mode: str = "native", push=None, beta: float = 0.0,
                         box=None, landmarks5=None, identity_box=None):
        anonymizer = _get_anonymizer(identity_path)
        if anonymizer is False:
            return {"path": None, "reason": "identity_unusable"}
        if not _select_source(anonymizer, identity_path, identity_box):
            return {"path": None, "reason": "identity_wrong_face"}
        _apply_mode(anonymizer, identity_path, mode, push, beta)
        image = cv2.imread(crop_path)
        if image is None:
            raise RuntimeError(f"could not read {crop_path}")
        try:
            if box is not None and landmarks5 is not None:
                result_frame = _swap_given_face(anonymizer, image, box, landmarks5)
            else:
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
        cv2.imwrite(str(out_path), result_frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])   # read once by the caller
        return {"path": str(out_path), "reason": None}

    serve(args.socket, {"swap": swap, "swap_with_reason": swap_with_reason, "check_identity": check_identity})


if __name__ == "__main__":
    main()
