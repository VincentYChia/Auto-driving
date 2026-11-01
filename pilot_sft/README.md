# Pilot fine-tuning: a first test model (2026-11-01)

A small Qwen3.5 vision-language model, fine-tuned with LoRA to write the next 4 s of ego path as
text: 8 "(x, y)" positions, one every 0.5 s. This is a **test of the data and the pipeline**, not the
research model. The research model (masked-diffusion action head, codebook tokens, C2) comes later
and reuses the same canonical record.

Evidence marks as in `docs/RESEARCH_DIRECTION.md`: **V** checked against a primary source or measured
here, **R** reported by others, **D** derived by us. Everything here is [proposed].

## What goes in and out
- **Data:** the `pilot_v1` view (`data_pipeline/v1/pilot_view.py`, plan step 7): `train.jsonl`,
  `dev.jsonl`, and the CAM_F0 images downscaled to 640 × 360.
- **Prompt:** speed, acceleration, yaw rate, past positions at −1.5 / −1.0 / −0.5 s, and the route command.
  These are only test-time inputs.
- **Answer:** `(6.6, 0.1) (13.2, 0.1) … (48.7, 0.2)`, metres, x forward, y left.
- **Score:** open-loop ADE / FDE on the dev split (NAVSIM's val logs), next to the constant-velocity
  baseline on the same samples. These are not NAVSIM scores.

| Measured on this computer, 2026-11-01 (V) | |
|---|---|
| Tokens per sample (Qwen3.5 tokenizer, image included) | **472**, of which 220 are image tokens |
| Chat export for Together.ai (base64 image inlined) | 67 KB per sample: 0.7 GB for the smoke set, ~3.0 GB for v1-pilot (D) |
| Constant-velocity baseline, smoke dev (1,000) | ADE **2.09 m**, FDE **4.97 m** |
| Stock `qwen3.5:2b` in Ollama, zero-shot, 50 dev samples | 38 of 50 answers lack 8 positions; the other 12: ADE 7.99 m vs 2.61 m for constant velocity |
| Qwen3.5-0.8B LoRA, RTX 3050 8 GB, batch 4 | 0.53 samples/s (252 tokens/s), peak 7.8 GB. Dev loss 0.64 → 0.32 in 20 steps |

On Windows, Qwen3.5's linear-attention layers fall back to slow reference kernels
(`flash-linear-attention` and `causal_conv1d` are not installed). On Linux GPUs, install them.

## Where to run it, and what it costs
Prices were checked 2026-11-01 (V, the providers' pricing pages). Run sizes come from the measured
472 tokens per sample (D).

| Where | Model | What you do | Smoke set (10k, 2 epochs) | v1-pilot (45k, 2 epochs) |
|---|---|---|---|---|
| **This computer** (RTX 3050 8 GB) | Qwen3.5-0.8B; the 2B is unlikely to fit at batch 4 (untested) | `train_lora.py` in `../train-venv` | ~10 h, ~$0.30 of electricity (D, at 0.2 kW and $0.16/kWh, assumed) | ~47 h, ~$1.50 (D) |
| **Anvil**, `gpu` partition (A100 40 GB) | Qwen3.5-2B (or 4B) | `scripts/anvil/pilot_sft/train_lora.sbatch` | ~1–3 GPU-hours | ~5–10 GPU-hours for 2B, ~10–20 for 4B |
| | | Cost | **$0**: 1 SU per GPU-hour (V); an ACCESS Explore allocation is ~6,000 GPU-hours (V, Purdue's example) | **$0** |
| **Together.ai** fine-tuning | **Qwen3.5-4B** (its smallest vision model; the 2B is not offered, V) | upload `chat_train.jsonl`, start a LoRA job | 9.4M tokens → **$4** (the per-job minimum, V) | 42.5M tokens → **~$15–17**; 3 epochs ~$22–25 |
| **Rented GPU** (RunPod A100 80 GB $1.19–1.79/h; H100 $1.99–3.99/h, V) | Qwen3.5-2B or 4B | copy the view (~0.6 / 2.4 GB), run `train_lora.py` | ~$2–5 | **~$6–18** for 2B; ~$12–36 for 4B |
| **Ollama** | — | **Cannot fine-tune (V).** Use it for zero-shot baselines (done above) and for serving | — | — |

How the estimates are built (D unless marked):
- **Together.ai** bills (epochs × training tokens) + (evaluations × validation tokens), at $0.34–0.38
  per million tokens for Qwen3.5 up to 9B, with a $4 minimum per job (V).
  - Its docs don't say whether image tokens are billed. If they are not, the cost roughly halves.
  - **Testing the model costs extra.** Either serve the adapter on a dedicated endpoint (H100 $5.49/h,
    V; ~$5 for the 1,000 dev samples), or download it (`checkpoint=adapter`, V) and score it on Anvil.
    A 4B model in bf16 does not fit this computer's 8 GB.
- **GPU-hours** assume 1,500–3,500 tokens/s for a 2B LoRA on one A100. The text-only reference is
  3,521 tokens/s for Gemma-2B LoRA on an A100 40 GB (V, LlamaFactory paper); the vision layers and
  padding make ours slower.
  - **Not measured yet.** Run the 200-step timing job on `gpu-debug` before trusting these figures.
- **For scale, the research model (LLaDA-V 8B, MVLAD-AD):**
  - The roadmap estimates ~60–120 H100-hours for 45k samples at 8 epochs (D, scaled from
    `docs/ROADMAP.md`): ~$160–480 rented, or an Anvil AI (H100) allocation.
  - Together.ai cannot train it. Custom models must be plain causal LMs without remote code (V).
  - Ollama cannot run it, and llama.cpp has no LLaDA-V support (V).

**Recommendation [proposed]:**
1. **Free first run:** Qwen3.5-2B on Anvil, if the allocation is active.
2. **No allocation?** Qwen3.5-4B on Together.ai: ~$4 for the smoke set, ~$16 for v1-pilot, plus ~$5 to test it.
3. **Run the smoke set first** wherever you train. It tells you whether the model beats constant
   velocity before you pay for the full set.

## How to run
**Environment.** Python 3.10+, torch with CUDA, then `pip install -r pilot_sft/requirements.txt`
(transformers 5; Qwen3.5 needs it, V). This computer has one in `..\train-venv`.

**This computer or a rented GPU:**
```bash
python pilot_sft/train_lora.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke \
    --model Qwen/Qwen3.5-0.8B --out ../runs/qwen35-0.8b-smoke --epochs 1
python pilot_sft/evaluate.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke \
    --model Qwen/Qwen3.5-0.8B --adapter ../runs/qwen35-0.8b-smoke/adapter --samples 500 \
    --out ../runs/qwen35-0.8b-smoke/eval.json
```
`--max-steps 200` makes a timing run.

**Anvil:**
1. Create a conda env `pilot` with the requirements.
2. Copy the view: `rsync -a data/pretrain/v1.0-pilot/views/pilot_v1/ anvil:$TRAINING_DATA_ROOT/pilot_v1/`.
3. Submit the jobs (header of `scripts/anvil/pilot_sft/train_lora.sbatch`):
   - **Timing first:** `SET=smoke MAX_STEPS=200 sbatch --time=00:30:00 --partition=gpu-debug ...`
   - **Then the run:** `SET=v1-pilot sbatch ...`

**Together.ai:**
1. `python -m data_pipeline.v1.pilot_view export --out data/pretrain/v1.0-pilot --set smoke`
   writes `chat_train.jsonl` / `chat_dev.jsonl` in the format its vision fine-tuning takes (V):
   OpenAI `messages`, with the image as a base64 data URL in the user turn.
2. Upload the file and start a LoRA job on Qwen3.5-4B, with `chat_dev.jsonl` as the validation file.
   - Use their CLI or the web UI. Check `together models list` for the model's exact name.
   - Expect the $4 minimum for the smoke set.
3. Download the adapter, then score it with `evaluate.py --adapter` on a GPU that fits 4B (Anvil).

**Ollama (zero-shot, local):**
```bash
ollama pull qwen3.5:2b
python pilot_sft/evaluate.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke --backend ollama --model qwen3.5:2b --samples 50
```

## Rules that still apply
- Train on `train.jsonl` only. Pick checkpoints on `dev.jsonl`. Never use navtest / navhard / warmup.
- The images and the exported files stay off GitHub (`.gitignore`, plan §5). Uploading them to a
  provider for training is a use of nuPlan data under CC BY-NC-SA. The researcher decides that (D10).
