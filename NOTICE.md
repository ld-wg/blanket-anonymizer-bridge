# NOTICE — third-party provenance and compliance flags

This repository's own purpose is exactly to isolate the following
third-party code/models from `lose-the-faces-keep-the-lesson`'s own
process — see that project's `src/pipeline/phase2_generate/models/blanket/NOTICE.md`
for the full architectural reasoning. This file covers what's specific to
*this* repo: the actual provenance of what's vendored/depended on here.

## Provenance — BLANKET

- **Source repo:** https://github.com/ctu-vras/blanket-infant-face-anonym
- **License: GPL-3.0.**
- **Vendored as a git submodule** at `vendor/blanket-infant-face-anonym/`,
  pinned to commit `84245d76dc7dc1cc59c5b356f1530346d422b922` (`master`
  HEAD at fetch time, 2026-09-24, committed 2025-12-17 — the repo's own
  README's "Dec 2025: Cleaner version of the code available" release).
- **Paper:** Hadera, Čech, Purkrabek, Hoffmann, *BLANKET: Anonymizing
  Faces in Infant Video Recordings*, IEEE ICDL 2025, pp. 1–8.
  arXiv: 2512.15542.

## Provenance — FaceFusion (vendored inside BLANKET's own repo)

- **Source repo:** https://github.com/facefusion/facefusion
- **License: OpenRAIL-AS** (a "responsible AI license" with real
  use-restriction clauses — read `vendor/blanket-infant-face-anonym/external/facefusion/LICENSE.md`
  directly, don't just cite the name).
- **Copyright:** (c) 2025 Henry Ruhs.
- Included as a full copy inside BLANKET's own repo
  (`external/facefusion/`), not a submodule of BLANKET's — this repo
  depends on it transitively, through the BLANKET submodule, without
  modifying it.

## Compliance flag 1 — GPL-3.0 scope of this repo

This repo is licensed GPL-3.0 because `identity_server.py`/`swap_server.py`
import BLANKET's own GPL-3.0 code directly (an in-process import, the real
combination GPL-3.0's copyleft is about) — unlike
`lose-the-faces-keep-the-lesson`, which only ever spawns this repo's
scripts as separate OS processes over a Unix socket. That process boundary
is a widely-used practice for keeping a GPL dependency's copyleft from
propagating into a differently-licensed or unlicensed calling project, but
**it is not a legal certainty** — the FSF's own guidance on this depends on
how tightly coupled the communication is. **This is not legal advice.**
Get real compliance/legal sign-off before any use beyond a personal or
academic research context.

## Compliance flag 2 — `inswapper_128` provenance

`swap_server.py` uses FaceFusion's `inswapper_128` face-swap model
(BLANKET's own configured default — see `facefusion_parameters.yaml` in
the submodule). This checkpoint has a well-known, unresolved provenance/
distribution controversy: the original author restricted official
redistribution after misuse concerns (non-consensual deepfakes), and its
training data provenance was never publicly disclosed. Community mirrors
(including FaceFusion's own hosting/download infrastructure) continue to
redistribute it, but its licensing/distribution status remains disputed.
This is inherited from BLANKET/FaceFusion, not introduced here, but
flagged in full because this repo is choosing to depend on it. Needs its
own compliance/ethics review before any use beyond personal/academic
research.

## Compliance flag 3 — FaceFusion's OpenRAIL-AS use restrictions

Read `vendor/blanket-infant-face-anonym/external/facefusion/LICENSE.md`'s
actual clauses before relying on this bridge for anything beyond the
anonymization/privacy-protection research purpose it exists for. OpenRAIL
licenses carry real behavioral-use restrictions, not just attribution
requirements — don't treat the license name as a formality.

## Real upstream bug found and worked around (2026-09-24)

`StableDiffusionAnonymizer.__init__`'s own default `config_path` resolution
(used whenever `config_path=None`) is off by one directory level:
`Path(__file__).parent.parent / "configs" / "module_parameters" / "stable_diffusion_parameters.yaml"`
resolves to `blanket/anonymization/configs/module_parameters/...`, which
does not exist — confirmed via a real `FileNotFoundError` on first
construction. The actual file lives at
`blanket/configs/module_parameters/stable_diffusion_parameters.yaml`.
BLANKET's own `generate_synthetic_identity()` never hits this because it
always computes and passes the *correct* path itself (three parents up
from `image_pipeline.py`, not `stable_diffusion.py`'s own two) before
constructing `StableDiffusionAnonymizer` — this bug is latent in normal
upstream usage and only surfaces because `identity_server.py` constructs
`StableDiffusionAnonymizer` directly (see README for why). Worked around
by passing the correct `config_path` explicitly, not by patching their
source.

## Real upstream bug #2 found and worked around (2026-09-24)

`blanket/core/detectors/detector_factory.py` resolves its own config files
via bare relative `Path` objects (`Path("blanket/configs/detector_parameters/...")`),
not derived from `__file__` — assumes the process's current working
directory IS BLANKET's own repo root (true when running their own
`run_video.py`/`run_image_anonymization.py` from inside that checkout;
false for a server script living in this sibling repo). Confirmed via a
real `FileNotFoundError` on the very first `generate` RPC. Both
`identity_server.py` and `swap_server.py` now `os.chdir()` to the BLANKET
submodule root at startup — matching the CWD their own code implicitly
expects, not a source patch. (Confirmed necessary for `identity_server.py`;
added to `swap_server.py` too as a precaution, not yet independently
hit there.)

## Real diffusers-version incompatibility found and worked around (2026-09-24)

Both `StableDiffusionAnonymizer._load_pipeline()` (base/ControlNet inpaint
stage) AND `SDRefiner.load()` (refiner stage) unconditionally call
`<pipeline>.enable_vae_slicing()` — confirmed crashing on real `generate()`
calls with `AttributeError: '...Pipeline' object has no attribute
'enable_vae_slicing'`, first on `StableDiffusionXLControlNetInpaintPipeline`
(reached only after successfully downloading/loading all 7 base pipeline
components), then again on `StableDiffusionXLImg2ImgPipeline` (the
refiner) once the first was patched — a genuine, apparently version-wide
gap across SDXL pipeline classes in the installed `diffusers` version, not
specific to one class or a download/config problem.
`enable_sequential_cpu_offload()`/`enable_attention_slicing(1)`, called
just before it in both places, work fine on both classes.
`identity_server.py` patches this externally, on each class (not
BLANKET's source files): `enable_vae_slicing` is set to delegate to the
VAE's own real `.enable_slicing()` if missing, for
`StableDiffusionXLControlNetInpaintPipeline`,
`StableDiffusionXLImg2ImgPipeline`, and (defensively, not yet hit)
`StableDiffusionXLInpaintPipeline` — preserving the actual memory-saving
behavior (not just no-op'ing it away), which matters given serra1's GPUs
are shared and already under real memory pressure from another user's job.

## Real upstream bug #4 found and worked around (2026-09-24): refiner resolution mismatch

Both `generate_synthetic_identity()` and `StableDiffusionAnonymizer.generate()`
pass `output_size=(orig_w, orig_h)` — the INPUT crop's own raw
dimensions — straight through to the SDXL refiner stage
(`SDRefiner.refine()`). Confirmed crashing with `ValueError: operands
could not be broadcast together with shapes (128,112,3) (133,117,1)`
inside `refine()`'s own final compositing line
(`refined_array * mask_array + base_array * (1.0 - mask_array)`): the
refiner's own `StableDiffusionXLImg2ImgPipeline` doesn't reliably preserve
odd/small input resolutions — our face crops are far smaller and less
"round" than BLANKET's own square demo images (which never hit this),
so its own output silently ends up a few pixels off from the mask saved
before the refiner ran. `identity_server.py` works around this by
generating at BLANKET's own native, config-declared resolution (896×896
by default, read from `anonymizer.config`, not hardcoded) instead of the
crop's own dimensions, then resizing the result (and its mask) back down
to the crop's size itself, before the Poisson-blend compositing step —
not a source patch, just not feeding their refiner a resolution it
doesn't handle correctly, and doing the resize-back ourselves since
BLANKET's own code never needed to (it always assumed `output_size`
already matched the input image).

## Real finding, not a bug: identity generation can produce a face FaceFusion's own detector can't find (2026-09-24)

Confirmed via a real end-to-end run: one identity's generated synthetic
face (an extreme down/side head angle, inherited faithfully by BLANKET's
own ControlNet conditioning from the original crop's own pose) had no
face `FaceFusionDirectAnonymizer.__init__`'s own `face_analyser` could
detect — it raises `ValueError: No face detected in source: ...` at
construction time. This is a real quality/robustness limitation of
BLANKET's own pipeline on non-infant, non-frontal real-world footage, not
an integration bug — this project's own faces are school-age children/
adults in a classroom setting, not the controlled infant recordings
BLANKET was evaluated on. `swap_server.py` now catches this specific
`ValueError` and treats it as a legitimate "nothing to swap" result
(caching the failure per `identity_path`, since retrying the same
identity image would fail identically every time, and
`FaceFusionDirectAnonymizer.__init__`'s own expensive `pre_check()` calls
already ran before the check that raises this) — the affected track
passes through unmodified for its whole duration, same contract as any
other "no usable face" case.

## Deliberate deviation from BLANKET's own prompt: dropped "baby" (2026-09-24)

BLANKET's own real prompt (`stable_diffusion_parameters.yaml`) is
literally `"high quality photo of a baby face, ..."`. This project's own
first real-video run found generated identities skewing strongly
infant-like regardless of the actual subject's age. Verified the
checkpoint itself (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) is
a generic public SDXL inpainting release, not fine-tuned on an infant
dataset — the bias traces to this one prompt phrase, not the model
weights, so no different checkpoint is needed. `identity_server.py`'s
`--prompt` now defaults to the same wording with "a baby face" replaced by
"a person's face" (every other quality/style descriptor kept verbatim from
BLANKET's own prompt); pass BLANKET's original wording via `--prompt` to
A/B against it directly. `--negative-prompt` still defaults to BLANKET's
own real value, unchanged. Not yet wired through
`lose-the-faces-keep-the-lesson`'s own `run.py` CLI — currently only
overridable by invoking `identity_server.py` directly with a different
`--prompt`.

## Real upstream bug #5 found and worked around (2026-09-24): seamlessClone crash near frame edges

Confirmed crashing on `video-demo-2.mov` (denser scene, faces closer to
frame edges than `video-demo.mov`): `cv2.seamlessClone` requires the
mask's own bounding region, placed at its centroid, to stay fully inside
the destination image — a face near the crop's own edge (itself near the
source frame's edge, since `crop_box()` clamps there) can produce a mask
whose extent pokes past the crop, raising a hard `cv2.error`. Neither this
project's port of `generate_synthetic_identity()`'s Poisson-blend step nor,
as far as verified, BLANKET's own real function guards against this — a
latent bug in the mechanism itself, never triggered by BLANKET's own
centered square demo images. `identity_server.py` now catches `cv2.error`
here and falls back to a plain hard-mask paste — the same fallback
pattern this project's own `models/_compositing.py::poisson_composite()`
already uses for the identical real-world failure mode, not a new one.

## Real upstream bug #6 found and worked around (2026-09-24): IoU-filter rejection crash

Confirmed crashing the whole pipeline run on `video-demo-2.mov`, at
frame 100: `FaceFusionDirectAnonymizer.anonymize()` raises
`RuntimeError("IoU filter rejected all faces - use previous frame")` when
`iou_filter` (real, on by default) rejects every detected face against
`previous_bboxes` for a given identity — a legitimate, expected outcome
in BLANKET's own real pipeline (their own `VideoPipeline.run()` catches
this and reuses the last successful frame), but `swap_server.py` only
caught "No faces detected" before this fix, so it propagated uncaught.
Now caught alongside "No faces detected", both treated as a transient
per-frame "nothing to swap" (not cached — unlike the "no face in identity
image" case, this is not a permanent property of the identity, the very
next frame gets a fresh attempt) and passed through as `null`, matching
this project's own existing no-usable-face contract rather than adding
new per-identity "last successful swap" state to replicate BLANKET's own
exact fallback.

## Upstream config values that are silently ignored or clamped (found 2026-09-24)

Found while reading the code paths this bridge calls, to plan the calling
project's identity-guidance work. Neither causes a crash; both mean
BLANKET's shipped configuration does not do what it says. This bridge
leaves both unchanged, so the baseline matches upstream behaviour.

1. **The `scheduler` key is never read.**
   `blanket/configs/module_parameters/stable_diffusion_parameters.yaml:19`
   says `scheduler: DPMSolverMultistepScheduler`, but no Python file under
   `blanket/` contains the string `scheduler` (grep on the pinned
   submodule). The pipeline keeps the scheduler from the checkpoint's own
   `scheduler_config.json`: `EulerDiscreteScheduler`, epsilon prediction.
   Anyone reproducing BLANKET from the YAML alone would pick the wrong
   sampler.
2. **`face_swapper_weight = 100` is outside the valid range and clamps.**
   BLANKET's `blanket/anonymization/methods/facefusion.py:143` sets
   `state_manager.init_item('face_swapper_weight', 100)`. FaceFusion's own
   range is 0.0–1.0 (`face_swapper/choices.py:25`, and the CLI enforces it
   through `choices=`; `init_item` bypasses that check).
   `balance_source_embedding` (`face_swapper/core.py:699-710`) maps the
   weight with `numpy.interp(weight, [0, 1], [0.35, -0.35])`, and `interp`
   clamps, so `w = -0.35`. The swap is conditioned on
   `1.35 · source − 0.35 · target`, where `target` is the embedding of the
   real face **in the current frame**. In effect, BLANKET pushes the swap
   away from the real person by a fixed amount, one frame at a time. That
   may be intended, but the value suggests a 0–100 scale was assumed.

**The second point's mix is between two different spaces (measured
2026-09-25).** For `inswapper`, `prepare_source_embedding`
(`core.py:686-691`) projects the source through the model's `emap`
initializer and divides by the raw embedding's norm, while
`balance_source_embedding` only L2-normalizes the target, with no `emap`
projection. `tools/check_embedding_space.py`, run on 300 faces from
`video-demo-2.mov`, measured how far apart those spaces are:

- **`emap` is far from orthogonal.** ‖EᵀE − I‖_F / √512 = 51.87, and its
  singular values run from 0.0013 to 19.84.
- **`e·E` points in an unrelated direction.** cos(e, e·E) over real ArcFace
  embeddings: mean 0.0014 (min −0.085, max 0.095). The projected source
  and the unprojected target are effectively orthogonal spaces.

So BLANKET's native `1.35·(e_src·E) − 0.35·ê_target` does not push the
swap away from the real face in the swapper's conditioning space. It adds
a vector that, in that space, bears no relation to the target identity.
This mix is FaceFusion's general `face_swapper_weight` behavior for
inswapper, not something BLANKET introduced; BLANKET's out-of-range value
only pins it at the extreme. The calling project's `track` swap mode
pushes in raw ArcFace space *before* the projection. Because the
projection is linear, that is a push away from p·E in the conditioning
space, which is the consistent version.

The same run also confirmed that FaceFusion's
`.assets/models/arcface_w600k_r50.onnx` (sha256 `f1f79dc3…70d1`) and
insightface buffalo_l's `w600k_r50.onnx` (sha256 `4c06341c…9e43`) are
different files but the same model: cosine 1.0000 on all 300 identical
aligned crops. The calling project's track-level identity estimate
therefore lives in the swap stage's own embedding space.

## Known open risks (not yet resolved empirically)

- **`requirements-identity.txt`'s plain `torch`/`torchvision` pin was
  CONFIRMED FAILING on serra1 (2026-09-24)**: a plain `pip install -r
  requirements-identity.txt` started downloading `torch==2.14.0` +
  `nvidia_cudnn_cu13`/`nvidia_nccl_cu13` (CUDA 13.x runtime) before being
  caught and killed partway through — same class of failure this
  project's own calling repo already solved for its torch dependency.
  Fixed by adding `--extra-index-url https://download.pytorch.org/whl/cu124`
  and pinning the exact `torch==2.6.0+cu124`/`torchvision==0.21.0+cu124`
  versions already verified working on this same machine. Re-verify if
  either pin is ever bumped.
- **Same class of bug found again, one layer deeper (2026-09-24)**:
  `transformers` eagerly imports `torchaudio` via a transitive chain
  (`transformers.audio_utils`, reached through its CLIP image-processor
  import path — imported even for pure-vision use, no audio anywhere in
  this pipeline) whenever it's installed at all. `torchaudio` wasn't
  pinned in this file, so pip resolved its own latest (`2.11.0`, CUDA
  13-targeted `libcudart.so.13`) — confirmed breaking `import diffusers`
  with the exact same `OSError: libcudart.so.13: cannot open shared
  object file` symptom as the torch/torchvision bug above. Fixed the same
  way: pinned `torchaudio==2.6.0+cu124` through the same index.
- **`requirements-swap.txt` deliberately pins `onnxruntime-gpu<1.21`**,
  older than FaceFusion's own vendored `requirements.txt` (`onnxruntime==1.23.2`,
  and that one's CPU-only anyway) — chosen to match
  `lose-the-faces-keep-the-lesson`'s own proven-working CUDA-12.5-ceiling
  range on serra1, NOT verified yet against FaceFusion's actual code at
  this specific version. First real thing to check once `.venv-swap` is
  installed and `swap_server.py` is actually run.
- **`requirements-swap.txt` excludes `gradio`/`gradio-rangeslider`**
  (FaceFusion's own UI-only dependencies) — not verified whether any
  core module this bridge imports (`state_manager`, `face_analyser`,
  `face_swapper`, `face_enhancer`) eagerly imports UI code at module load
  time. If the first real server boot raises `ImportError` for gradio,
  add it back — that's an expected, cheap fix, not a sign of a design
  problem.
- **Python ≥3.10 is required** (BLANKET's own `pyproject.toml`/`setup.py`
  both declare the looser `>=3.9`, but its vendored FaceFusion hard-checks
  `sys.version_info < (3, 10)` at runtime) — `setup.sh` defaults to
  `python3.12` to stay well clear of this.

## Calibration / setup log

Empty as of repo creation — entries go here once real installs/runs have
happened on serra1, dated, same discipline as the calling project's own
NOTICE.md files (a real, tested finding, not a guess).
