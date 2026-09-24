# blanket-anonymizer-bridge

Thin, unmodified process bridge around [BLANKET](https://github.com/ctu-vras/blanket-infant-face-anonym)
(Hadera, Čech, Purkrabek, Hoffmann, *BLANKET: Anonymizing Faces in Infant
Video Recordings*, IEEE ICDL 2025, arXiv:2512.15542), used as a Phase 2
face-generation backend by [lose-the-faces-keep-the-lesson](https://github.com/ld-wg/lose-the-faces-keep-the-lesson),
a video face-anonymization research pipeline (undergraduate thesis
project).

## What this is

Two small server scripts (`identity_server.py`, `swap_server.py`), each
importing one specific real building block of BLANKET's own pipeline and
exposing it over a tiny Unix-socket JSON protocol (see `rpc_server.py`),
so the calling project never has to import BLANKET's code directly:

- **`identity_server.py`** wraps `blanket.anonymization.pipelines.image_pipeline`'s
  own Stable-Diffusion-XL + dual-ControlNet + refiner identity generation.
- **`swap_server.py`** wraps `blanket.anonymization.methods.facefusion.FaceFusionDirectAnonymizer`,
  BLANKET's own real, vendored FaceFusion-based face-swap step.

Each runs in its own separate Python virtual environment (`.venv-identity`,
`.venv-swap`, created by `setup.sh`) — the two stages have almost entirely
disjoint dependencies (torch/diffusers vs. onnxruntime/FaceFusion), and
splitting them avoids pulling BLANKET's own TensorFlow/Flask/Gradio
dependency chain (needed by *other* parts of its own repo this integration
never calls) into either one.

**Nothing here reimplements or modifies BLANKET's own code.** Both server
scripts import BLANKET's real classes/functions and call them as-is
(with one narrow, documented exception — see below); BLANKET's own repo is
included as a pinned git submodule under `vendor/blanket-infant-face-anonym/`.

## Why a separate repository

This repo exists specifically so BLANKET's GPL-3.0 code never gets imported
into `lose-the-faces-keep-the-lesson`'s own process — the two projects
communicate only through OS-level mechanisms (a Unix socket + temp files),
which is why the calling project's own `models/blanket/backend.py` only
ever spawns this repo's two scripts as subprocesses, never `import`s
anything from here. Since this repo *does* import BLANKET directly, it is
itself licensed **GPL-3.0** (see `LICENSE`), kept as its own repository so
that license boundary is legible and auditable rather than buried in an
undocumented folder on a shared server. Full reasoning:
`lose-the-faces-keep-the-lesson`'s own `src/pipeline/phase2_generate/models/blanket/NOTICE.md`.

**This is not a resolved legal question** — see `NOTICE.md`'s compliance
flags before using this beyond a personal/academic research context.

## One documented deviation from calling BLANKET's own functions directly

`identity_server.py` does not call `image_pipeline.generate_synthetic_identity()`
directly — that function constructs a fresh `StableDiffusionAnonymizer`
(reloading the entire SDXL pipeline) and calls `.unload()` on every single
invocation, which would defeat the whole point of a persistent server.
Instead, `identity_server.py` constructs one `StableDiffusionAnonymizer`
at startup and re-implements that function's own surrounding logic (face/
landmark detection, a seed-override workaround — see NOTICE.md — and the
same Poisson-blend post-step, reading the same config keys BLANKET's own
YAML already sets) around that one warm instance. The actual diffusion
call itself (`anonymizer.generate(...)`) is still BLANKET's own real,
unmodified method.

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd blanket-anonymizer-bridge
./setup.sh            # creates .venv-identity, .venv-swap; installs each
                       # role's hand-picked requirements-*.txt
```

Requires Python ≥3.10 (see `NOTICE.md` for why BLANKET's own declared
`>=3.9` is not actually sufficient). `setup.sh` defaults to `python3.12`;
pass a different interpreter path as its one argument if needed.

## License

GPL-3.0 (see `LICENSE`) — this repo imports BLANKET's GPL-3.0 code
directly, so that's the correct license for it. See `NOTICE.md` for the
full third-party provenance chain (BLANKET itself, its vendored FaceFusion
under OpenRAIL-AS, and `inswapper_128`'s own distribution/provenance
controversy) and the compliance flags that come with each.

## Purpose statement

This bridge exists to support face de-identification / privacy-protection
research on educational video (a school-age-children anonymization
pipeline), not to produce non-consensual likenesses of real people. If
you're evaluating whether to build on this, read FaceFusion's own
OpenRAIL-AS responsible-use clauses (linked from `NOTICE.md`) — they apply
to this bridge's use of it too.
