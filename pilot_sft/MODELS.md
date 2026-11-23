# Saved models

Every trained model is kept, in two places: the lab RTX 4090 desktop (`~/navsim_workspace/exp/`) and Anvil scratch (`$NAVSIM_EXP_ROOT`, `/anvil/scratch/x-vchia/navsim_workspace/exp/`). Researcher, 2026-11-19: "Make sure we save all the models too." Nothing is deleted; a model that is no longer used stays where it is. The weights are not in this repository, only this list.

## Rules

- **Pilot adapters** (`pilot_sft/<run>/adapter/`, and `epoch_<k>/` for every epoch but the last when the run snapshots): written on the machine that trained them, copied to the other one. Anvil's autopilot pulls each finished desktop run (all but `checkpoint.pt` and scratch images); runs trained on Anvil are copied to the desktop with `~/xv/to_desk_models.sh`.
- **Path generators (heads)** (`heads/<run>/`): every epoch's weights (`epoch_<k>.pt`), the final weights, the last full checkpoint (weights + optimizer, for a resume), `run.json` and `log.jsonl`. RL runs keep every evaluated checkpoint. Each finished run is copied to Anvil scratch.
- **Anvil scratch purges files that go unused.** Check the purge rule and the oldest files by about 2026-12-05. The desktop copy is the one that does not expire.
- **Checksums:** the first 16 hex digits of `adapter_model.safetensors`'s SHA-256 are the same on both machines for every adapter below (checked 2026-11-19).

## Pilot adapters (2026-11-19: 53 adapters, 11.5 GB)

S3's per-epoch NAVSIM runs (`qwen35-0.8b-v1fullthin-p12-visfull-ep4-e1` to `-e3`) link to S3's `epoch_1` to `epoch_3` and are not listed twice. The `test-*` and `timing-*` rows are timing and smoke runs, kept like the others.

| Run | Weights | MB | SHA-256 (first 16) | Desktop | Anvil |
|---|---|---|---|---|---|
| `Qwen3.5-2B-v1-large-21928579` | adapter | 155 | `3621f45c0ac28692` | yes | yes |
| `qwen35-0.8b-drop80` | adapter | 107 | `72f8f91ab765a80f` | yes | yes |
| `qwen35-0.8b-ep6` | adapter | 107 | `23a17aca10d646b1` | yes | yes |
| `qwen35-0.8b-large-ep3` | adapter | 107 | `28750515b71285b2` | yes | yes |
| `qwen35-0.8b-lr5e4` | adapter | 107 | `eab8775aa0f70f65` | yes | yes |
| `qwen35-0.8b-mini-hard-base-s0` | adapter | 107 | `fb77de5ab7239df0` | yes | yes |
| `qwen35-0.8b-mini-hard-base-s1` | adapter | 107 | `5bfe1ffeb8fbbd78` | yes | yes |
| `qwen35-0.8b-mini-hard-drop0` | adapter | 107 | `ec7dfad1d679d547` | yes | yes |
| `qwen35-0.8b-mini-hard-drop25` | adapter | 107 | `5520245f04b96d91` | yes | yes |
| `qwen35-0.8b-mini-hard-visfull` | adapter | 308 | `6a8e6e01798bcca0` | yes | yes |
| `qwen35-0.8b-mini-thin-base-s0` | adapter | 107 | `3e37c012a62c216f` | yes | yes |
| `qwen35-0.8b-mini-thin-base-s1` | adapter | 107 | `10cceff0a3995143` | yes | yes |
| `qwen35-0.8b-rank128` | adapter | 366 | `14d3a5488a96c101` | yes | yes |
| `qwen35-0.8b-v1fullthin-p12-visfull-ep4` | adapter | 308 | `1b9d9cea6bf66ef3` | yes | yes |
| `qwen35-0.8b-v1fullthin-p12-visfull-ep4` | epoch_1 | 308 | `cc2dc40479aff25a` | yes | yes |
| `qwen35-0.8b-v1fullthin-p12-visfull-ep4` | epoch_2 | 308 | `f3d021bd52cc40d4` | yes | yes |
| `qwen35-0.8b-v1fullthin-p12-visfull-ep4` | epoch_3 | 308 | `a5a084d842a80281` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep4` | adapter | 308 | `873ab2a451d13802` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep4` | epoch_1 | 308 | `be774e1886572f88` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep4` | epoch_2 | 308 | `655b315d6ebbf707` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep4` | epoch_3 | 308 | `d251e4c3cf300f5c` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | adapter | 308 | `1182c68de6653e30` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | epoch_1 | 308 | `a44e3d92f2403914` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | epoch_2 | 308 | `442e223c91845406` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | epoch_3 | 308 | `61f75b7dd63a8326` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | epoch_4 | 308 | `245a3aaa8fdc5c88` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | epoch_5 | 308 | `196b40f73db59fb5` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | epoch_6 | 308 | `8783ea8dd2c800db` | yes | yes |
| `qwen35-0.8b-v1hard-p12-visfull-ep8` | epoch_7 | 308 | `ba632cf057180c36` | yes | yes |
| `qwen35-0.8b-v1large-p12-ep3` | adapter | 107 | `87847cf58f2e9eb4` | yes | yes |
| `qwen35-0.8b-v1large-p12-ep3` | epoch_1 | 107 | `c9815aebbc5863a6` | yes | yes |
| `qwen35-0.8b-v1large-p12-ep3` | epoch_2 | 107 | `3fc7fee72e955bff` | yes | yes |
| `qwen35-0.8b-v1large-p12-unmasked-ep3` | adapter | 107 | `a5d03c225594072e` | yes | yes |
| `qwen35-0.8b-v1large-p12-unmasked-ep3` | epoch_1 | 107 | `c610f0865030eae0` | yes | yes |
| `qwen35-0.8b-v1large-p12-unmasked-ep3` | epoch_2 | 107 | `40e3efef9f116955` | yes | yes |
| `qwen35-0.8b-v1nt-p12-ep3` | adapter | 107 | `a2cbf3cf7c46dd30` | yes | yes |
| `qwen35-0.8b-v1nt-p12-ep3` | epoch_1 | 107 | `c3c0514eb96ecc3d` | yes | yes |
| `qwen35-0.8b-v1nt-p12-ep3` | epoch_2 | 107 | `b7b133e145402db0` | yes | yes |
| `qwen35-0.8b-v1pilot-egodrop50` | adapter | 107 | `38d34bb2377f3f13` | yes | yes |
| `qwen35-0.8b-xlnd-ep4s` | adapter | 107 | `ec3eb45e16a25ae1` | yes | yes |
| `qwen35-0.8b-xlnd-ep4s` | epoch_1 | 107 | `cf6efd66cdcc5281` | yes | yes |
| `qwen35-0.8b-xlnd-ep4s` | epoch_2 | 107 | `71fdc1b86cd8e90f` | yes | yes |
| `qwen35-0.8b-xlnd-ep4s` | epoch_3 | 107 | `0090043ddce9e90a` | yes | yes |
| `qwen35-0.8b-xlnd-visfull-ep4s` | adapter | 308 | `5e7b661062591f9f` | yes | yes |
| `qwen35-0.8b-xlnd-visfull-ep4s` | epoch_1 | 308 | `f1fb6f594159fa54` | yes | yes |
| `qwen35-0.8b-xlnd-visfull-ep4s` | epoch_2 | 308 | `d190c41c1ed132bc` | yes | yes |
| `qwen35-0.8b-xlnd-visfull-ep4s` | epoch_3 | 308 | `51ffda98e73c5f1c` | yes | yes |
| `qwen35-2b-v1pilot-egodrop50` | adapter | 155 | `a85154b3d82b4ee4` | yes | yes |
| `test-vision-cpu` | adapter | 509 | `3b0fccbe6376d49a` | yes | yes |
| `test-vision-cpu` | epoch_1 | 509 | `52263cd601b3a4e3` | yes | yes |
| `test-vision-full` | adapter | 308 | `a8bf862e1cadff5e` | yes | yes |
| `test-vision-full` | epoch_1 | 308 | `a85e3040aeb12a35` | yes | yes |
| `timing-2b-v1pilot` | adapter | 155 | `9af42b5e5e9b0c85` | yes | yes |

## Path generators

Each training run's folder (`exp/heads/<run>/`) holds `epoch_1.pt` … `epoch_20.pt`, `final.pt` (= epoch 20), `checkpoint.pt` (with the optimizer), `run.json`, `log.jsonl` and its prediction folders. Configurations are defined in `pilot_sft/EXPERIMENTS.md`, "Definitions". The SHA-256 is of `final.pt`.

| Run | Configuration | Stored layers | Seed | Epochs | Parameters | Training time | GB | SHA-256 of final.pt (first 16) | Desktop | Anvil |
|---|---|---|---|---|---|---|---|---|---|---|
| `a1-l24` | A1 (masked diffusion, 0.1 m bins, distance-aware embeddings) | 24 | 0 | 20 | 41.1M | 40 min | 3.95 | `49f6607d62c9b157` | yes | yes |
| `a1-l12-18-24` | A1 | 12, 18, 24 | 0 | 20 | 41.9M | 2 h 20 min | 4.33 | `9aeefcba1ab7fd86` | yes | yes |
| `d-l24` | D (anchored diffusion, 20 k-means anchors) | 24 | 0 | 20 | 40.3M | 40 min | 3.8 | `ecd8c713d579e5b2` | yes | yes |
| `d-l12-18-24` | D | 12, 18, 24 | 0 | 20 | 41.1M | 2 h 18 min | 3.9 | `3ec3fea646824f78` | yes | yes |
| `b-l24` | B (DDIM, predicts the clean path) | 24 | 0 | 20 | 40.2M | 47 min | 4.0 | `36268012f4f5179f` | yes | yes |
| `b-l12-18-24` | B | 12, 18, 24 | 0 | 20 | 41.0M | 2 h 3 min | 3.9 | `afa0797e37e1d862` | yes | yes |
| `c-l24` | C (flow matching, start noise SD 0.1) | 24 | 0 | 20 | 40.2M | 40 min | 3.9 | `d55bcfc919735b40` | yes | yes |
| `c-l12-18-24` | C | 12, 18, 24 | 0 | 20 | 41.0M | 1 h 43 min | 3.9 | `eca30a454091ef89` | yes | yes |
| `r-l24` | R (regression control, L1 on the normalised waypoints) | 24 | 0 | 20 | 40.2M | 39 min | 3.6 | `416805a4384dbb8d` | yes | yes |
| `r-l12-18-24` | R | 12, 18, 24 | 0 | 20 | 41.0M | 2 h 9 min | 3.7 | `b4743f835d991869` | yes | yes |
| `a1-l12-18-24-s1` | A1 (phase 3, seed 1) | 12, 18, 24 | 1 | 20 | 41.9M | 1 h 55 min | 3.8 | `a94e915be08f507b` | yes | yes |
| `a1-l12-18-24-s2` | A1 (phase 3, seed 2) | 12, 18, 24 | 2 | 20 | 41.9M | 2 h 2 min | 3.8 | `c1ad469de6b2144a` | yes | yes |
| `b-l12-18-24-s1` | B (phase 3, seed 1) | 12, 18, 24 | 1 | 20 | 41.0M | 1 h 44 min | 3.7 | `07530ac1c7f30fc8` | yes | yes |
| `b-l12-18-24-s2` | B (phase 3, seed 2) | 12, 18, 24 | 2 | 20 | 41.0M | 1 h 53 min | 3.7 | `3480cb13e2eef3b3` | yes | yes |
| `c-l12-18-24-s1` | C (phase 3, seed 1) | 12, 18, 24 | 1 | 20 | 41.0M | 1 h 44 min | 3.7 | `eff2b9bb010c3156` | yes | yes |
| `c-l12-18-24-s2` | C (phase 3, seed 2) | 12, 18, 24 | 2 | 20 | 41.0M | 1 h 40 min | 3.7 | `91e92e7dd868aa44` | yes | yes |
| `d-l12-18-24-s1` | D (phase 3, seed 1) | 12, 18, 24 | 1 | 20 | 41.1M | 2 h 17 min | 3.7 | `133e7eaade275145` | yes | yes |
| `d-l12-18-24-s2` | D (phase 3, seed 2) | 12, 18, 24 | 2 | 20 | 41.1M | 1 h 53 min | 3.7 | `0d5e2ae9c30a6f10` | yes | yes |
| `r-l12-18-24-s1` | R (phase 3, seed 1) | 12, 18, 24 | 1 | 20 | 41.0M | 1 h 53 min | 3.7 | `1f33cf93477a9e9b` | yes | yes |
| `r-l12-18-24-s2` | R (phase 3, seed 2) | 12, 18, 24 | 2 | 20 | 41.0M | 1 h 43 min | 3.7 | `8e4e97f287ecadda` | yes | yes |
| `selector-rehearsal-gen-half0` | A1 on drive half 0 (78 of 158 drives, 118,387 scenes), the selector chain's rehearsal (R14; used for no choice) | 12, 18, 24 | 0 | 20 | 41.9M | 1 h 2 min | 3.8 | `f729e5ca383cc97a` | yes | yes |
| `selector-rehearsal-gen-half1` | A1 on drive half 1 (80 of 158 drives, 117,746 scenes), the selector chain's rehearsal (R14; used for no choice) | 12, 18, 24 | 0 | 20 | 41.9M | 56 min | 3.8 | `b208c5241ddb36a2` | yes | yes |
| `a0-l12-18-24` | A0 (phase 2: plain bins, no notion of distance) | 12, 18, 24 | 0 | 20 | 42.4M | 1 h 55 min | 4.0 | `78d702da529bd222` | yes | yes |
| `a2-l12-18-24` | A2 (phase 2: A1 plus soft labels) | 12, 18, 24 | 0 | 20 | 41.9M | 1 h 50 min | 3.9 | `dfc01c8e9f19811d` | yes | yes |
| `a3-l12-18-24` | A3 (phase 2: A2 plus the position-error loss and the averaged readout) | 12, 18, 24 | 0 | 20 | 41.9M | 2 h 16 min | 4.0 | `f5106da50c3d7c46` | yes | yes |
| `a4-l12-18-24` | A4 (phase 2: A3 plus the continuous offset within the bin) | 12, 18, 24 | 0 | 20 | 41.9M | 2 h 5 min | 4.1 | `de6e237b596c9be5` | yes | yes |
| `d-r150-l12-18-24` | D, R15 tuning run 0 (phase 6): 10 anchors | 12, 18, 24 | 0 | 20 | 41.1M | 2 h 33 min | 3.8 | `ef3ae700c843d6fb` | yes | yes |
| `d-r151-l12-18-24` | D, R15 tuning run 1 (phase 6): 40 anchors | 12, 18, 24 | 0 | 20 | 41.1M | 2 h 10 min | 3.8 | `513e96ecfc97fda2` | yes | yes |
| `d-r152-l12-18-24` | D, R15 tuning run 2 (phase 6): anchor truncation 0.02 (`--anchor-trunc 0.02`) | 12, 18, 24 | 0 | 20 | 41.1M | 2 h 12 min | 3.8 | `d0e2f65e7a06d177` | yes | yes |
| `d-r153-l12-18-24` | D, R15 tuning run 3 (phase 6): anchor truncation 0.1 (`--anchor-trunc 0.1`) | 12, 18, 24 | 0 | 20 | 41.1M | 1 h 56 min | 3.8 | `c53276f9051b0955` | yes | yes |
| `b-r150-l12-18-24` | B, R15 tuning run 0 (phase 6): predicts the noise (`--target eps`) | 12, 18, 24 | 0 | 20 | 41.0M | 1 h 51 min | 3.9 | `20808865328ae87a` | yes | yes |
| `b-r151-l12-18-24` | B, R15 tuning run 1 (phase 6): predicts v (`--target v`) | 12, 18, 24 | 0 | 20 | 41.0M | 2 h 22 min | 3.9 | `8a0a188cf4c133e7` | yes | yes |
| `b-r152-l12-18-24` | B, R15 tuning run 2 (phase 6): linear noise schedule (`--schedule linear`) | 12, 18, 24 | 0 | 20 | 41.0M | 2 h 7 min | 3.9 | `4bf8427e0e358139` | yes | yes |
| `b-r153-l12-18-24` | B, R15 tuning run 3 (phase 6): noise target and linear schedule (`--target eps --schedule linear`) | 12, 18, 24 | 0 | 20 | 41.0M | 1 h 54 min | 3.9 | `eba4daa59e682b4b` | yes | yes |
| `c-r150-l12-18-24` | C, R15 tuning run 0 (phase 6): flow sigma 0.3 (`--flow-sigma 0.3`) | 12, 18, 24 | 0 | 20 | 41.0M | 1 h 52 min | 3.8 | `804cdafe344f7ce9` | yes | yes |
| `c-r151-l12-18-24` | C, R15 tuning run 1 (phase 6): flow sigma 1.0 (`--flow-sigma 1.0`) | 12, 18, 24 | 0 | 20 | 41.0M | 2 h 8 min | 3.8 | `48579ecb78c4c2c1` | yes | yes |
| `c-r152-l12-18-24` | C, R15 tuning run 2 (phase 6): Beta time sampling (`--time-sampling beta`) | 12, 18, 24 | 0 | 20 | 41.0M | 2 h 12 min | 3.8 | `67b9a0445b79a200` | yes | yes |
| `c-r153-l12-18-24` | C, R15 tuning run 3 (phase 6): logit-normal time sampling (`--time-sampling logit-normal`) | 12, 18, 24 | 0 | 20 | 41.0M | 1 h 52 min | 3.8 | `c9ddde4d79ddefae` | yes | yes |

### RL runs

An RL run's folder (`exp/heads/<run>/`) holds `rl_<u>.pt` for every saved update `u` (the same format as a supervised `epoch_<k>.pt`, so `predict_head.py` loads either), `checkpoint.pt` (weights + optimizer at the last update), `run.json` (start weights, settings, RL scenes) and `log.jsonl` (one line per update). The SHA-256 is of each `rl_<u>.pt`.

| Run | Starts from | Recipe | RL scenes | Updates saved | GB | SHA-256 (first 16) | Desktop | Anvil |
|---|---|---|---|---|---|---|---|---|
| `a1-l24-rl-trial` | `a1-l24` final (epoch 20) | as registered: lr 1e-5, no KL, 8 candidates × 32 scenes | 2,662 (3 training drives) | 100, 200 | 0.79 | rl_100 `797714288468da1a`, rl_200 `30025febf573336c` | yes | yes |
| `a1-l12-18-24-rlrecipe0` | `a1-l12-18-24` epoch 20 | R12 recipe RL0 (as registered) | 80,626 (navtrain in our data) | 250, 500 | 0.81 | rl_250 `3a0d63cf9db4f15c`, rl_500 `b7dc9b3a7217bf3e` | yes | yes |
| `a1-l12-18-24-rlrecipe1` | `a1-l12-18-24` epoch 20 | R12 RL1: lr 1e-5, KL β 0.04, 8 × 32 | 80,626 | 250, 500 | 0.82 | rl_250 `76bbe925ba243f10`, rl_500 `0d6deb00ce921927` | yes | yes |
| `a1-l12-18-24-rlrecipe2` | `a1-l12-18-24` epoch 20 | R12 RL2: lr 3e-6, KL β 0.04, 8 × 32 | 80,626 | 250, 500 | 0.82 | rl_250 `f035f5484bd22147`, rl_500 `6b81a968e1e72ca2` | yes | yes |
| `a1-l12-18-24-rlrecipe3` | `a1-l12-18-24` epoch 20 | R12 RL3: lr 3e-6, KL β 0.04, 16 × 16 | 80,626 | 250, 500 | 0.81 | rl_250 `9bed347cf859ab87`, rl_500 `289906f0920dab69` | yes | yes |
| `d-l12-18-24-rlrecipe0` | `d-l12-18-24` epoch 5 | R12 RL0 (as registered) | 80,626 | 250, 500 | 0.81 | rl_250 `b09f696a9e726178`, rl_500 `56be595268e47e5c` | yes | yes |
| `d-l12-18-24-rlrecipe1` | `d-l12-18-24` epoch 5 | R12 RL1 | 80,626 | 250, 500 | 0.81 | rl_250 `a6981d1dc8c37fc6`, rl_500 `3dc03640e2c86393` | yes | yes |
| `d-l12-18-24-rlrecipe2` | `d-l12-18-24` epoch 5 | R12 RL2 | 80,626 | 250, 500 | 0.81 | rl_250 `a11aec0e814bc9a5`, rl_500 `0b3b86d5b99c84cf` | yes | yes |
| `d-l12-18-24-rlrecipe3` | `d-l12-18-24` epoch 5 | R12 RL3 | 80,626 | 250, 500 | 0.78 | rl_250 `784fb59d4c021807`, rl_500 `1ae831b1d42ef387` | yes | yes |
| `a1-l12-18-24-rlrecipe4` | `a1-l12-18-24` epoch 20 | R13 RL4: advantage = reward − group mean (Dr. GRPO), lr 1e-5, no KL, 8 × 32 | 80,626 | 250, 500 | 0.81 | rl_250 `bf752057bfe54a96`, rl_500 `6003a877907cad50` | yes | yes |
| `d-l12-18-24-rlrecipe4` | `d-l12-18-24` epoch 5 | R13 RL4, with the 0.1 SD floor inside the log-probability | 80,626 | 250, 500 | 0.80 | rl_250 `b3e972c5e0bc37e6`, rl_500 `4da1a8feaccea0a6` | yes | yes |
| `a1-l12-18-24-rlrecipe5` | `a1-l12-18-24` epoch 20 | R13 RL5: GRPO advantage (divided by the group SD) + BC term, measured BC weight 2.50 | 80,626 | 250, 500 | 0.81 | rl_250 `8948041715be4d5e`, rl_500 `2a5640f5f980cddb` | yes | yes |
| `a1-l12-18-24-rlrecipe6` | `a1-l12-18-24` epoch 20 | R13 RL6: Dr. GRPO + BC term, BC weight 0.73 | 80,626 | 250, 500 | 0.81 | rl_250 `7fe787a0710c5fe3`, rl_500 `fdec139132a88809` | yes | yes |
| `d-l12-18-24-rlrecipe5` | `d-l12-18-24` epoch 5 | R13 RL5: GRPO advantage + BC term (weight 2.65), 0.1 SD floor | 80,626 | 250, 500 | 0.80 | rl_250 `481a28237eef9ea4`, rl_500 `4bc232f0e86ff9de` | yes | yes |
| `d-l12-18-24-rlrecipe6` | `d-l12-18-24` epoch 5 | R13 RL6: Dr. GRPO + BC term (weight 0.40), 0.1 SD floor | 80,626 | 250, 500 | 0.80 | rl_250 `1761e0354ad043d7`, rl_500 `ab9406c8ac13819c` | yes | yes |
| `d-l12-18-24-rlrecipe7` | `d-l12-18-24` epoch 5 | R13 RL7: the 0.1 SD floor only (GRPO advantage, no BC) | 80,626 | 250, 500 | 0.80 | rl_250 `378bb7bf97407aa2`, rl_500 `b9ef42b11ea34470` | yes | yes |
| `a1-l12-18-24-rl` | `a1-l12-18-24` epoch 20 | phase 4 (R16), RL4; navdev choice rl_1250 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1250 `5f739333b3f76e60`, rl_1500 `d3a706b03b71df77` | yes | yes |
| `a1-l12-18-24-s1-rl` | `a1-l12-18-24-s1` epoch 20 | phase 4 (R16), RL4; navdev choice rl_1500 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1500 `eda77894a526c5e5` | yes | yes |
| `a1-l12-18-24-s2-rl` | `a1-l12-18-24-s2` epoch 20 | phase 4 (R16), RL4; navdev choice rl_750 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_750 `250520dfabfb6429`, rl_1500 `ba135a5df5c7f91f` | yes | yes |
| `a1-l12-18-24-rl6` | `a1-l12-18-24` epoch 20 | phase 4 (R16), RL6; navdev choice rl_1500 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1500 `bcd4cbb164009611` | yes | yes |
| `a1-l12-18-24-s1-rl6` | `a1-l12-18-24-s1` epoch 20 | phase 4 (R16), RL6; navdev choice rl_1250 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1250 `10d112a6bbe74b96`, rl_1500 `548bc08c6254119e` | yes | yes |
| `a1-l12-18-24-s2-rl6` | `a1-l12-18-24-s2` epoch 20 | phase 4 (R16), RL6; navdev choice rl_1500 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1500 `26e60263948585f0` | yes | yes |
| `b-l12-18-24-rl` | `b-l12-18-24` epoch 10 | phase 4 (R16), RL4; navdev choice rl_750 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_750 `4d7df9eb8e1328b6`, rl_1500 `750d4a0f39cd4a2e` | yes | yes |
| `b-l12-18-24-rl6` | `b-l12-18-24` epoch 10 | phase 4 (R16), RL6; navdev choice rl_1250 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1250 `17689ea421f9a446`, rl_1500 `51cd4db3ccb65d5d` | yes | yes |
| `c-l12-18-24-rl` | `c-l12-18-24` epoch 10 | phase 4 (R16), RL4; navdev choice rl_750 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_750 `d7641cac3c6372e4`, rl_1500 `4ea9ab6fdf911db2` | yes | yes |
| `c-l12-18-24-rl6` | `c-l12-18-24` epoch 10 | phase 4 (R16), RL6; navdev choice rl_1250 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1250 `8ae2c4d567d97d92`, rl_1500 `c5f9af9a413797da` | yes | yes |
| `d-l12-18-24-rl` | `d-l12-18-24` epoch 5 | phase 4 (R16), RL4; navdev choice rl_1000 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1000 `6fc70cbaab1ed06d`, rl_1500 `032f93092f183ce2` | yes | yes |
| `d-l12-18-24-rl6` | `d-l12-18-24` epoch 5 | phase 4 (R16), RL6; navdev choice rl_1500 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1500 `14aef4ae40630ae8` | yes | yes |
| `d-l12-18-24-s1-rl` | `d-l12-18-24-s1` epoch 5 | phase 4 (R16), RL4; navdev choice rl_250 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_250 `7910608ddeacdf88`, rl_1500 `a47244b1515630f1` | yes | yes |
| `d-l12-18-24-s1-rl6` | `d-l12-18-24-s1` epoch 5 | phase 4 (R16), RL6; navdev choice rl_1500 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1500 `f6671039579493b4` | yes | yes |
| `d-l12-18-24-s2-rl` | `d-l12-18-24-s2` epoch 5 | phase 4 (R16), RL4; navdev choice rl_500 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_500 `f60746416418cd09`, rl_1500 `d57ef5d789dbeb76` | yes | yes |
| `d-l12-18-24-s2-rl6` | `d-l12-18-24-s2` epoch 5 | phase 4 (R16), RL6; navdev choice pending (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1500 `069f84747b09fb44` (all six checked, 2026-11-22 14:13) | yes | yes |

A's RL7 is the same run as its RL0 (A's steps are not Gaussian, so the SD floor does nothing). All R12 and R13 recipe runs are on both machines with matching checksums (2026-11-20 10:49).

The RL trial lowered navdev's single-path PDMS (81.18 → 73.97 after 200 updates), which led to R12 in `docs/HEADS_PLAN.md`. It is kept as the evidence for that rule, not used further.
