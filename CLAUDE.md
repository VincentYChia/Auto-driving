# Notes for Claude sessions in this repo

This is a fork of MVLAD-AD (masked-diffusion driving VLA, LLaDA-V 8B). The researcher is extending it toward an **edge-deployable** masked-diffusion driving VLA evaluated on **NAVSIM**.

## Read first
1. `docs/RESEARCH_DIRECTION.md`: the source of truth for claims, the novelty check, edge-compute limits, open decisions and the decision log.
2. `docs/ROADMAP.md`: the work plan, the workstreams table and GPU (training) compute.
3. Then the workstream you are working on:
   - **A. NAVSIM evaluation:** `docs/navsim_eval/RUNBOOK.md`.
   - **B. Training data:** `docs/training_data/README.md`. Its process, principles and open decisions D1–D11 govern all data work.

## Workstreams are separate
| | A. NAVSIM evaluation | B. Training data |
|---|---|---|
| Code | `navsim_eval/` | `data_pipeline/v1/` (first build); top-level modules are prototype v0.1 |
| Jobs | `scripts/anvil/navsim_eval/` | `scripts/local/` (default); `scripts/anvil/data_pipeline/` |
| Tests | `tests/eval/` | `tests/data/` |

- **Shared:** `navsim_tools/` (read-only NAVSIM access: `geometry`, `raw_frames` are pure numpy; `loader` needs the devkit), `tests/shared/`, `scripts/anvil/*.sh`.
- **No cross-imports:** neither workstream imports the other.
- **Training data is deliberate.** Months of research depend on it. Don't add data features, change the schema, or run full extraction or teacher labelling unless the researcher has signed off the relevant phase in `docs/training_data/README.md`.
  - The prototype jobs refuse full runs unless `ALLOW_FULL_RUN=1`.
  - Propose; don't rush.

## Tests
- `python -m pytest tests` runs the pure tests anywhere.
- `tests/*/test_*integration.py` run only where `navsim` and `nuplan` are importable (Python 3.9 with NAVSIM's pins, e.g. numpy 1.23.4). They are skipped otherwise.

## Keeping alignment
- When the direction changes, edit `RESEARCH_DIRECTION.md` and add a dated row to its decision log.
- Mark items **[agreed]** only when the researcher has confirmed them. Research suggestions stay **[proposed]**.
- The researcher wants data-backed answers. Mark evidence **V** (verified against a primary source), **R** (self-reported or snippet only) or **D** (derived). Never present an estimate as a measurement. Prefer official docs over recall.

## Repo pitfalls (details in `docs/ROADMAP.md`, "Known problems")
- The Git LFS weights (adapters, `tokenizer/tokenizer_model.pth`) return 404 from this fork; the files present are pointer stubs. Run `scripts/download_mvlad_weights.sh` (official repo, sha256-verified). Don't commit the downloaded files.
- `train/scripts/stage*.sh` call `llava.train.train_mem`, which doesn't exist. The action-token training code is in `llava/train/train.py`.
- The waypoint-token count (10) is hardcoded in training, inference and decoding. The planning prompt differs between training and inference.
- The shipped image setting (`anyres_max_4`) gives ~3.6k image tokens per frame.

## Environment facts
- **Data builds (small scale):** run on the researcher's own computer, CPU only, writing to `data/pretrain/` in the repo [agreed 2026-10-29]. Only ID lists, manifests and aggregate statistics are committed. Fine-tuning compute is open: a cloud GPU or Anvil.
- **Training/eval GPUs:** Purdue Anvil. `gpu` partition = A100 40 GB; 48 h jobs; 12 GPUs per user. Anvil AI (H100) needs a separate allocation. No CPU allocation. Scratch is purged 30 days after last access.
- **On-device latency claims** must be measured on a physical Jetson (AGX Thor primary, AGX Orin stretch). Anvil is not a proxy.
- **NAVSIM rule:** never train on navtest / navhard / warmup splits.
