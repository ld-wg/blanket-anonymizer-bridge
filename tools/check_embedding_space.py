#!/usr/bin/env python3
"""Embedding-space check for the swap stage (calling project's contribution, Step 4a).

Run with the swap venv: `.venv-swap/bin/python tools/check_embedding_space.py --images '<glob>'`.
Prints aggregate numbers only (no per-face embedding is printed or written).

Answers two questions before `swap_server.py`'s `track` mode is trusted:

1. Is FaceFusion's recognizer (`.assets/models/arcface_w600k_r50.onnx`)
   the same model as insightface buffalo_l's `w600k_r50.onnx`, which the
   calling project uses to estimate each track's real identity? Both files
   are hashed, and both models embed the *same* FaceFusion-aligned 112x112
   crops (template `arcface_112_v2`, `/127.5 − 1`, BGR→RGB), so any
   difference comes from the model file alone.
2. How close to orthogonal is inswapper's `emap` (the last initializer of
   `inswapper_128.onnx`, which `prepare_source_embedding` multiplies the
   source embedding by)? BLANKET's native mix adds an emap-projected
   source to a non-projected target; that is only consistent if emap
   roughly preserves directions. Reported: ‖EᵀE − I‖_F / √512 and
   cos(e, eE/‖eE‖) over the real embeddings.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_BLANKET_ROOT = _HERE / "vendor" / "blanket-infant-face-anonym"
sys.path.insert(0, str(_BLANKET_ROOT))
sys.path.insert(0, str(_BLANKET_ROOT / "external" / "facefusion"))
os.chdir(_BLANKET_ROOT)  # same working directory swap_server.py uses

_CONFIG_PATH = _BLANKET_ROOT / "blanket" / "configs" / "module_parameters" / "facefusion_parameters.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stats(values) -> str:
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    return (f"n={v.size} mean={v.mean():.4f} min={v.min():.4f} p5={np.percentile(v, 5):.4f} "
            f"median={np.median(v):.4f} max={v.max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="glob of images containing faces")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--insightface-onnx", default=os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx"))
    args = parser.parse_args()

    import cv2
    import numpy as np
    import onnxruntime as ort

    from blanket.anonymization.methods.facefusion import FaceFusionDirectAnonymizer
    from facefusion import face_analyser
    from facefusion.face_helper import warp_face_by_face_landmark_5
    from facefusion.model_helper import get_static_model_initializer
    from facefusion.processors.modules.face_swapper import core as face_swapper

    paths = sorted(glob.glob(args.images))
    random.Random(0).shuffle(paths)
    if not paths:
        sys.exit(f"no images match {args.images}")

    # Constructing BLANKET's own anonymizer initializes every FaceFusion
    # state item and downloads/pre-checks the models, exactly as the swap
    # server does. The first image that has a face serves as its "identity".
    anonymizer = None
    for p in paths:
        try:
            anonymizer = FaceFusionDirectAnonymizer(synthetic_face_path=p, config_path=str(_CONFIG_PATH))
            break
        except ValueError:
            continue
    if anonymizer is None:
        sys.exit("no face found in any image")

    ff_model = _BLANKET_ROOT / "external" / "facefusion" / ".assets" / "models" / "arcface_w600k_r50.onnx"
    print(f"facefusion recognizer   {ff_model}  sha256={_sha256(ff_model)}")
    print(f"insightface w600k_r50   {args.insightface_onnx}  sha256={_sha256(Path(args.insightface_onnx))}")

    session = ort.InferenceSession(args.insightface_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    swapper_path = face_swapper.get_model_options().get("sources").get("face_swapper").get("path")
    emap = get_static_model_initializer(swapper_path).astype(np.float64)
    print(f"inswapper               {swapper_path}  emap shape={emap.shape}")

    same_model_cos, emap_cos = [], []
    for p in paths:
        if len(same_model_cos) >= args.limit:
            break
        image = cv2.imread(p)
        if image is None:
            continue
        faces = face_analyser.get_many_faces([image])
        if not faces:
            continue
        face = faces[0]
        crop, _ = warp_face_by_face_landmark_5(image, face.landmark_set.get("5/68"), "arcface_112_v2", (112, 112))
        blob = (crop / 127.5 - 1)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
        e_if = session.run(None, {input_name: blob})[0].ravel()
        e_ff = np.asarray(face.embedding, dtype=np.float64)
        same_model_cos.append(float(e_if @ e_ff / (np.linalg.norm(e_if) * np.linalg.norm(e_ff))))
        projected = e_ff @ emap
        emap_cos.append(float(e_ff @ projected / (np.linalg.norm(e_ff) * np.linalg.norm(projected))))

    gram = emap.T @ emap
    print(f"faces used              {len(same_model_cos)}")
    print(f"cos(facefusion, insightface) on identical crops: {_stats(same_model_cos)}")
    print(f"emap: ||E^T E - I||_F / sqrt(512) = {np.linalg.norm(gram - np.eye(gram.shape[0])) / np.sqrt(gram.shape[0]):.4f}"
          f"   singular values min={np.linalg.svd(emap, compute_uv=False).min():.4f}"
          f" max={np.linalg.svd(emap, compute_uv=False).max():.4f}")
    print(f"cos(e, e·E) over real embeddings: {_stats(emap_cos)}")


if __name__ == "__main__":
    main()
