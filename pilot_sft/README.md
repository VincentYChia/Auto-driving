# Pilot fine-tuning: a first test model (2026-11-01)

A small Qwen3.5 vision-language model, fine-tuned with LoRA to write the next 4 s of ego path as
text: 8 "(x, y)" positions, one every 0.5 s. This is a **test of the data and the pipeline**, not the
research model. The research model (masked-diffusion action head, codebook tokens, C2) comes later
and reuses the same canonical record.

Evidence marks as in `docs/RESEARCH_DIRECTION.md`: **V** checked against a primary source or measured
here, **R** reported by others, **D** derived by us. Everything here is [proposed].

## The two data sets
| | Smoke set | v1-pilot (the full set) |
|---|---|---|
| Samples | 10,000 train + 1,000 dev | 45,000 train + 2,000 dev |
| Logs | 130, from the first 20 camera archives | 1,304, from all 200 archives |
| Native images kept locally | 2.3 GB | 9.9 GB |
| Training view (640 × 360) | 0.55 GB, **ready** | ~2.4 GB (D), after the full download |
| Together.ai upload | 0.75 GB, **ready** | ~3.2 GB (D) |
| Built by | the commands in `PRETRAIN_V1_PLAN.md`, "How to run steps 4–7" (done) | `bash scripts/local/build_v1_pilot.sh` after the full fetch: canonical, QA page, view, chat export with Together's check, Anvil tar |

The smoke set is the small first sample, used to check the data and the training before spending on
the full set. The two overlap: 1,472 frames are in both (11,000 + 47,000 − 56,528).

## What goes in and out
- **Data:** the `pilot_v1` view (`data_pipeline/v1/pilot_view.py`, plan step 7): `train.jsonl`,
  `dev.jsonl`, and the CAM_F0 images downscaled to 640 × 360.
- **Prompt:** speed, acceleration, yaw rate, past positions at −1.5 / −1.0 / −0.5 s, and the route command.
  These are only test-time inputs.
- **Yaw-rate leak, fixed 2026-11-03 (prompt `pilot_v1.1`):** `pilot_v1.0` took the yaw rate from the
  heading change between −0.5 s and **+0.5 s**, so it carried half a second of the future, which
  NAVSIM's agent input does not have. `pilot_v1.1` uses −0.5 s to now. On the 2,000 v1-pilot dev samples
  the no-image bars move from 0.73 to 0.78 m (MLP), 0.87 to 0.93 (linear), 1.08 to 1.15 (kinematic) (V).
  The two smoke models below were trained and scored on `pilot_v1.0`, so their numbers are slightly
  optimistic. The v1-pilot runs on Anvil use `pilot_v1.1`.
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

### First test model: Qwen3.5-0.8B, smoke set, 1 epoch (2026-11-02, V)
500 dev samples (seed 0). Aggregates in `results/qwen35-0.8b-smoke-e1.json`.

| | ADE (m) | FDE (m) |
|---|---|---|
| Constant velocity | 2.20 | 5.23 |
| Kinematic (constant acceleration + yaw rate), no image | 1.26 | 3.40 |
| Linear regression on the prompt's numbers, no image | 0.99 | 2.70 |
| **Fine-tuned Qwen3.5-0.8B** | **0.97** | **2.58** |
| Same model, camera image blanked | 1.01 | 2.67 |
| Same model, ego state and past positions hidden | 8.24 | 14.55 |
| MLP on the prompt's numbers, no image (3-seed ensemble) | **0.82** | **2.24** |

- **The image is almost unused:** blanking it costs only 0.04 m.
- **The past positions are ignored too:** a fake sideways drift in them carries into the prediction at −0.01 (the no-image MLP: 0.93).
- **So the model drives from speed, acceleration and yaw rate.** The bar for the next model is the no-image MLP at 0.82 m. Hiding ego state during training (`--ego-dropout`) is the first thing to try [proposed].

### Second test model: ego state hidden on half the training samples
Same settings plus `--ego-dropout 0.5`, same 500 dev samples. Aggregates in `results/qwen35-0.8b-smoke-egodrop50.json`.

| ADE (m) | First model | `--ego-dropout 0.5` |
|---|---|---|
| Full input | 0.97 | 1.02 |
| Camera image blanked | 1.01 (+0.04) | 1.10 (+0.08) |
| Ego state and past positions hidden | 8.24 | **4.19** |
| Both hidden | — | 13.69 |
| No-image bar with ego state hidden: mean training path per route command | 6.31 | 6.31 |

- **The model can now read the image:** with ego state hidden it beats the command-mean bar by 2.1 m,
  and blanking the image as well takes it to 13.7 m.
- **It still doesn't use the image when the ego state is there:** blanking it costs 0.08 m, and
  the past positions are still ignored (drift follow −0.07).
- **The full-input error is 0.05 m worse**, and both models lose to the no-image MLP (0.82 m).
- For the v1-pilot run, ego dropout is a trade: a fallback that reads the image, for a small loss
  on full input. Whether more data makes the model use the image with ego state present is the
  open question the 45k run answers [proposed].

### v1-pilot: 45,000 samples, 2 epochs, `--ego-dropout 0.5` (2026-11-04, V)
1,000 dev samples (seed 0), prompt `pilot_v1.1`, one A100 on Anvil. Aggregates in
`results/qwen35-0.8b-v1pilot-egodrop50.json` and `results/qwen35-2b-v1pilot-egodrop50.json`.

| ADE / FDE (m) | Qwen3.5-0.8B | Qwen3.5-2B |
|---|---|---|
| **Full input** | **0.661 / 1.737** | **0.655 / 1.708** |
| Camera image blanked | 0.937 / 2.549 | 0.885 / 2.372 |
| Ego state and past positions hidden | 2.910 / 5.365 | 2.950 / 5.436 |
| Both hidden | 13.096 / 23.244 | 13.096 / 23.244 |
| Training time, peak GPU memory | 1.8 h, 13.8 GB | 2.6 h, 16.9 GB |

No-image bars on the same samples (ADE / FDE): constant velocity 1.890 / 4.509, kinematic 1.164 / 3.084,
linear 0.925 / 2.492, MLP 0.781 / 2.118, command mean 6.419 / 11.548.

- **Both models beat the no-image MLP** (by 0.12 m ADE), which no smoke-set model did.
- **They now use the image with ego state present:** blanking it costs 0.28 m (0.8B) and 0.23 m (2B), against 0.08 m on the smoke set.
- **With ego state hidden they beat the command-mean bar by 3.5 m.**
- **With nothing to go on they predict "stopped":** 13.096 / 23.244 is exactly the score of an all-zero path on these samples (V), so both models match.
- **The past positions are still ignored:** drift follow −0.01 for both, 0.37 for the MLP.
- **The two models are tied:** paired 2B − 0.8B on the same samples is −0.007 m ADE (95% bootstrap CI −0.028 to +0.014) and −0.028 m FDE (−0.087 to +0.031). NAVSIM decides between them.

### v1-large: 100,000 samples, 1 epoch, `--ego-dropout 0.5` (2026-11-05, V)
Qwen3.5-2B on the same 1,000 dev samples. Aggregates in `results/qwen35-2b-v1large-egodrop50.json`.

| ADE / FDE (m) | 45,000 samples, 2 epochs | 100,000 samples, 1 epoch |
|---|---|---|
| **Full input** | **0.655 / 1.708** | **0.657 / 1.708** |
| Camera image blanked | 0.885 / 2.372 | 0.876 / 2.339 |
| Ego state and past positions hidden | 2.950 / 5.436 | 2.950 / 5.420 |
| Samples seen in training | 90,016 | 100,000 |

- **2.2 times the distinct samples changed nothing measurable:** paired large − 45k is +0.003 m ADE (95% bootstrap CI −0.013 to +0.019) and −0.001 m FDE (−0.043 to +0.043).
- **Samples seen were about the same,** so this tests more variety, not more training. It does not test 2 epochs on the larger set.
- **The MLP's drift follow is not stable:** 0.37 on v1-pilot, 0.05 on v1-large. The pilot's stays at −0.01 to −0.02.

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
- **Checkpoints:** `train_lora.py` saves `<out>/checkpoint.pt` every 50 steps (`--save-every`).
- **Resuming:** `--resume` continues an interrupted run from that checkpoint, e.g. after a shutdown or an Anvil time limit. Tested 2026-11-02: an interrupted and resumed run logs the same losses as an uninterrupted one.
- **Cleanup:** the checkpoint is deleted once the adapter is saved.

**Is the model reading the road?** Score it against the no-image baselines, then probe it, on the
same dev samples (same `--samples` and `--seed` as `evaluate.py`):
```bash
python pilot_sft/baselines.py --view <view> --out ../runs/baselines-smoke.json      # bars to beat without the image
python pilot_sft/probes.py --view <view> --model <model> --adapter <run>/adapter --out <run>/probes.json
```
- **`probes.py`** reports ADE with the image blanked, with ego state and past positions hidden, and with both. Compare the ego-hidden score with `command_mean` from `baselines.py`.
- **`follow`** is how much of a fake sideways drift in the past positions the model carries forward: 1 = pure extrapolation, 0 = ignores it.
- **`--ego-dropout 0.5`** in `train_lora.py` hides ego state and past positions on half the training samples, so the answer has to come from the image. On the smoke set it taught image reading but cost 0.05 m on full input (results above) [proposed]. `EGO_DROPOUT=0.5` passes it on Anvil.

**Anvil: quick start.** Ready on this computer: `data/pretrain/upload/pilot_v1_smoke.tar` (549 MB:
images, jsonl, view.json). Rebuild it any time with `bash scripts/local/pack_pilot_view.sh smoke`.
```bash
# 1. On this computer: copy the packed view to your Anvil home
scp data/pretrain/upload/pilot_v1_smoke.tar <user>@anvil.rcac.purdue.edu:
# 2. On Anvil, once: get the code and the environment (login node, ~15 min)
git clone -b claude/stoic-darwin-ue0jm6 https://github.com/VincentYChia/Auto-driving.git ~/Auto-driving && cd ~/Auto-driving
export SBATCH_ACCOUNT=<your GPU allocation>          # see `mybalance`
bash scripts/anvil/pilot_sft/setup_env.sh
source scripts/anvil/common.sh && mkdir -p $TRAINING_DATA_ROOT/pilot_v1 && tar -xf ~/pilot_v1_smoke.tar -C $TRAINING_DATA_ROOT/pilot_v1
# 3. Timing run (30 min, debug partition), then the run; each job scores itself on dev at the end
SET=smoke MAX_STEPS=200 EVAL_SAMPLES=40 sbatch --time=00:30:00 --partition=gpu-debug scripts/anvil/pilot_sft/train_lora.sbatch   # gpu-debug: 30 min max
SET=smoke EPOCHS=2 sbatch scripts/anvil/pilot_sft/train_lora.sbatch
```
- **Output:** `$NAVSIM_EXP_ROOT/pilot_sft/<model>-<set>-<job>/`, holding `run.json` (speed, memory), `log.jsonl` and `eval.json`.
- **Other models:** `MODEL=Qwen/Qwen3.5-4B` trains the 4B instead.
- **Where things go:** the defaults in `scripts/anvil/config.sh` put data and runs on `$SCRATCH`, which is purged 30 days after last access.

**Together.ai: quick start.** Ready on this computer: `chat_train.jsonl` (682 MB) and `chat_dev.jsonl`
(66 MB) in `data/pretrain/v1.0-pilot/views/pilot_v1/smoke/`. Both pass Together's own file check
(SDK 3.0.0, V). `pilot_sft/together_finetune.py` holds the whole flow, using the SDK's real parameter names.
```bash
# in ../train-venv (has the together SDK) or after `pip install together`
export TOGETHER_API_KEY=...                               # PowerShell: $env:TOGETHER_API_KEY="..."
python pilot_sft/together_finetune.py launch --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke         # upload + price estimate, no charge
python pilot_sft/together_finetune.py launch --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke --yes   # start the job
python pilot_sft/together_finetune.py status   <job id>
python pilot_sft/together_finetune.py download <job id> --dest ../runs/together/<job id>
```
- **Settings [proposed]:** Qwen/Qwen3.5-4B (V), LoRA rank 32, loss on the answer only, vision encoder frozen, 2 epochs, learning rate 1e-4, 4 evaluations on dev. Batch size is Together's: 8 for this model (V).
- **Uploads:** the first `launch` uploads ~0.75 GB.
- **Scoring:** 4B doesn't fit this computer's GPU. Score the downloaded adapter on Anvil or a rented GPU with `evaluate.py --model Qwen/Qwen3.5-4B --adapter ...`, or on a Together dedicated endpoint.

**Ollama (zero-shot, local):**
```bash
ollama pull qwen3.5:2b
python pilot_sft/evaluate.py --view data/pretrain/v1.0-pilot/views/pilot_v1/smoke --backend ollama --model qwen3.5:2b --samples 50
```

## Rules that still apply
- Train on `train.jsonl` only. Pick checkpoints on `dev.jsonl`. Never use navtest / navhard / warmup.
- The images and the exported files stay off GitHub (`.gitignore`, plan §5). Uploading them to a
  provider for training is a use of nuPlan data under CC BY-NC-SA. The researcher decides that (D10).
