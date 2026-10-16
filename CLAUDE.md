# Notes for Claude sessions in this repo

This is a fork of MVLAD-AD (masked-diffusion driving VLA, LLaDA-V 8B). The researcher is extending it toward an **edge-deployable** masked-diffusion driving VLA evaluated on **NAVSIM**.

## Read first
1. `docs/RESEARCH_DIRECTION.md`: the source of truth for claims, the novelty check, edge-compute limits, open decisions and the decision log.
2. `docs/ROADMAP.md`: the work plan and GPU (training) compute.

## Keeping alignment
- When the direction changes, edit `RESEARCH_DIRECTION.md` and add a dated row to its decision log.
- Mark items **[agreed]** only when the researcher has confirmed them. Research suggestions stay **[proposed]**.
- The researcher wants data-backed answers. Mark evidence **V** (verified against a primary source), **R** (self-reported or snippet only) or **D** (derived). Never present an estimate as a measurement. Prefer official docs over recall.

## Repo pitfalls (details in `docs/ROADMAP.md`, "Known problems")
- The Git LFS weights (adapters, `tokenizer/tokenizer_model.pth`) return 404 from this fork; the files present are pointer stubs.
- `train/scripts/stage*.sh` call `llava.train.train_mem`, which doesn't exist. The action-token training code is in `llava/train/train.py`.
- The waypoint-token count (10) is hardcoded in training, inference and decoding. The planning prompt differs between training and inference.
- The shipped image setting (`anyres_max_4`) gives ~3.6k image tokens per frame.

## Environment facts
- **Training/eval GPUs:** Purdue Anvil. `gpu` partition = A100 40 GB; 48 h jobs; 12 GPUs per user. Anvil AI (H100) needs a separate allocation. No CPU allocation. Scratch is purged 30 days after last access.
- **On-device latency claims** must be measured on a physical Jetson (AGX Thor primary, AGX Orin stretch). Anvil is not a proxy.
- **NAVSIM rule:** never train on navtest / navhard / warmup splits.
