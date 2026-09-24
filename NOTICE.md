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
