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
| `selector-a4-gen-half0` | A3 on drive half 0 (78 of 158 drives, 118,387 scenes), contender A4's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.9M | 1 h 8 min | 3.7 | `bfdfe6fec41d0c50` | yes | yes |
| `selector-a4-gen-half1` | A3 on drive half 1 (80 of 158 drives, 117,746 scenes), contender A4's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.9M | 55 min | 3.7 | `84ad776e00077e36` | yes | yes |
| `selector-a6-gen-half0` | A3 on drive half 0 (78 of 158 drives, 118,387 scenes), contender A6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.9M | 1 h 10 min | 3.7 | `07de8b7d0cb609c0` | yes | yes |
| `selector-a6-gen-half1` | A3 on drive half 1 (80 of 158 drives, 117,746 scenes), contender A6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.9M | 55 min | 3.7 | `68ec72952fbf27b6` | yes | yes |
| `selector-b4-gen-half0` | B's phase-1 configuration on drive half 0 (78 of 158 drives, 118,387 scenes), contender B4's selector chain (R14, B4 deployed by R19); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 57 min | 3.7 | `90152c4abb3aa023` | yes | yes |
| `selector-b4-gen-half1` | B's phase-1 configuration on drive half 1 (80 of 158 drives, 117,746 scenes), contender B4's selector chain (R14, B4 deployed by R19); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 55 min | 3.7 | `8d624274592fd255` | yes | yes |
| `selector-b6-gen-half0` | B's tuned configuration (noise target, linear schedule) on drive half 0 (78 of 158 drives, 118,387 scenes), contender B6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 62 min | 3.7 | `493cb7162f011b8a` | yes | yes |
| `selector-b6-gen-half1` | B's tuned configuration (noise target, linear schedule) on drive half 1 (80 of 158 drives, 117,746 scenes), contender B6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 62 min | 3.7 | `d8c824969073a90e` | yes | yes |
| `selector-c4-gen-half0` | C's tuned configuration (Beta time sampling) on drive half 0 (78 of 158 drives, 118,387 scenes), contender C4's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 91 min (disk-bound) | 3.7 | `ccd7669a66489b6a` | yes | yes |
| `selector-c4-gen-half1` | C's tuned configuration (Beta time sampling) on drive half 1 (80 of 158 drives, 117,746 scenes), contender C4's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 58 min | 3.7 | `8bdb4a06a0e8ea14` | yes | yes |
| `selector-c6-gen-half0` | C's tuned configuration (Beta time sampling) on drive half 0 (78 of 158 drives, 118,387 scenes), contender C6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 63 min | 3.7 | `4e4383aeb0a224b5` | yes | yes |
| `selector-c6-gen-half1` | C's tuned configuration (Beta time sampling) on drive half 1 (80 of 158 drives, 117,746 scenes), contender C6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.0M | 55 min | 3.7 | `fa308442026a1f44` | yes | yes |
| `selector-d4-gen-half0` | D's phase-1 configuration on drive half 0 (78 of 158 drives, 118,387 scenes), contender D4's selector chain (R14); no RL, as D4 deploys epoch 5 without it: its `epoch_5.pt` (`1eb5459c94560055`) makes the candidates; navdev single path 72.51 (the deployed D4 77.10) | 12, 18, 24 | 0 | 20 | 41.1M | 58 min | 3.9 | `f9ef7d1f0633240d` | yes | yes |
| `selector-d4-gen-half1` | D's phase-1 configuration on drive half 1 (80 of 158 drives, 117,746 scenes), contender D4's selector chain (R14); no RL, as D4 deploys epoch 5 without it: its `epoch_5.pt` (`fd40d7b64e8326e2`) makes the candidates; navdev single path 72.01 (the deployed D4 77.10) | 12, 18, 24 | 0 | 20 | 41.1M | 56 min | 3.9 | `e90fcd9dae67455d` | yes | yes |
| `selector-d6-gen-half0` | D's tuned configuration (anchor truncation 0.1) on drive half 0 (78 of 158 drives, 118,387 scenes), contender D6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.1M | 59 min | 3.8 | `37277e92588030c5` | yes | yes |
| `selector-d6-gen-half1` | D's tuned configuration (anchor truncation 0.1) on drive half 1 (80 of 158 drives, 117,746 scenes), contender D6's selector chain (R14); its RL run below | 12, 18, 24 | 0 | 20 | 41.1M | 56 min | 3.8 | `a7f8a384fdd164d0` | yes | yes |
| `a0-l12-18-24` | A0 (phase 2: plain bins, no notion of distance) | 12, 18, 24 | 0 | 20 | 42.4M | 1 h 55 min | 4.0 | `78d702da529bd222` | yes | yes |
| `a2-l12-18-24` | A2 (phase 2: A1 plus soft labels) | 12, 18, 24 | 0 | 20 | 41.9M | 1 h 50 min | 3.9 | `dfc01c8e9f19811d` | yes | yes |
| `a3-l12-18-24` | A3 (phase 2: A2 plus the position-error loss and the averaged readout) | 12, 18, 24 | 0 | 20 | 41.9M | 2 h 16 min | 4.0 | `f5106da50c3d7c46` | yes | yes |
| `a4-l12-18-24` | A4 (phase 2: A3 plus the continuous offset within the bin) | 12, 18, 24 | 0 | 20 | 41.9M | 2 h 5 min | 4.1 | `de6e237b596c9be5` | yes | yes |
| `a3-no-metric-l12-18-24` | A3 without the metric embedding (phase 2: R17's leave-one-out from the series' best, A3) | 12, 18, 24 | 0 | 20 | 42.4M | 2 h 42 min | 4.1 | `63b660edd6f2ca8b` | yes | yes |
| `a3-l12-18-24-s1` | A3, seed 1 (phase 2: R17's seed of the series' best; also R20's first seed of contender A4) | 12, 18, 24 | 1 | 20 | 41.9M | 2 h 11 min | 3.8 | `f2f7bf8a13c8a1f9` | yes | yes |
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
| `d-l12-18-24-s2-rl6` | `d-l12-18-24-s2` epoch 5 | phase 4 (R16), RL6; navdev choice rl_1000 (R11) | 80,626 | 250-1,500 (6) | 1.5 | rl_1000 `a4e31030998362eb`, rl_1500 `069f84747b09fb44` (all six checked, 2026-11-22 14:13) | yes | yes |
| `a3-l12-18-24-rlfull4` | `a3-l12-18-24` epoch 5 | phase 6 (R16, R15, R18), full RL stage, RL4; round 1 of 5,000 updates; navdev choice rl_2500 (R11), so no round 2 | 80,626 | 500-5,000 (10) | 2.1 | rl_2500 `5625d995e59b0e46`, rl_5000 `bafb5dcd80ec827c` | yes | yes |
| `a3-l12-18-24-rlfull6` | `a3-l12-18-24` epoch 5 | phase 6 (R16, R15, R18), full RL stage, RL6; round 1 of 5,000 updates; navdev choice rl_3000 (R11), so no round 2 | 80,626 | 500-5,000 (10) | 2.1 | rl_3000 `572507dd3dbc2889`, rl_5000 `dae3d04847f369c6` | yes | yes |
| `selector-a4-gen-half0-rl` | `selector-a4-gen-half0` epoch 5 | phase 6 (R14), contender A4's half-0 generator: RL4 for 2,500 updates (as many as A4's deployed checkpoint) on half 0's RL scenes | 39,683 (half 0; half 1 has the other 40,943) | 500-2,500 (5) | 1.4 | rl_2500 `7258d77b7d59a348` | yes | yes |
| `selector-a4-gen-half1-rl` | `selector-a4-gen-half1` epoch 5 | phase 6 (R14), contender A4's half-1 generator: RL4 for 2,500 updates (as many as A4's deployed checkpoint) on half 1's RL scenes | 40,943 (half 1) | 500-2,500 (5) | 1.4 | rl_2500 `5929df65ea1ba7c9` | yes | yes |
| `selector-a6-gen-half0-rl` | `selector-a6-gen-half0` epoch 5 | phase 6 (R14), contender A6's half-0 generator: RL6 for 3,000 updates (as many as A6's deployed checkpoint) on half 0's RL scenes | 39,683 (half 0) | 500-3,000 (6) | 1.5 | rl_3000 `a61a162ae645902e` | yes | yes |
| `selector-a6-gen-half1-rl` | `selector-a6-gen-half1` epoch 5 | phase 6 (R14), contender A6's half-1 generator: RL6 for 3,000 updates (as many as A6's deployed checkpoint) on half 1's RL scenes | 40,943 (half 1) | 500-3,000 (6) | 1.5 | rl_3000 `116cddf1a9ce214b` | yes | yes |
| `selector-b4-gen-half0-rl` | `selector-b4-gen-half0` epoch 10 | phase 6 (R14), contender B4's half-0 generator: RL4 for 1,000 updates (as many as B4's deployed checkpoint) on half 0's RL scenes; navdev single path 83.24 (the deployed B4 85.30) | 39,683 (half 0) | 500-1,000 (2) | 0.9 | rl_1000 `ea48e488c854382a` | yes | yes |
| `selector-b4-gen-half1-rl` | `selector-b4-gen-half1` epoch 10 | phase 6 (R14), contender B4's half-1 generator: RL4 for 1,000 updates (as many as B4's deployed checkpoint) on half 1's RL scenes; navdev single path 84.61 (the deployed B4 85.30) | 40,943 (half 1) | 500-1,000 (2) | 0.9 | rl_1000 `973cf7143fac1d0d` | yes | yes |
| `selector-b6-gen-half0-rl` | `selector-b6-gen-half0` epoch 10 | phase 6 (R14), contender B6's half-0 generator: RL6 for 6,000 updates (as many as B6's deployed checkpoint) on half 0's RL scenes, 0 scorer errors; navdev single path 83.07 (the deployed B6 86.74) | 39,683 (half 0) | 500-6,000 (12) | 2.4 | rl_6000 `02dc7070dca799f0` | yes | yes |
| `selector-b6-gen-half1-rl` | `selector-b6-gen-half1` epoch 10 | phase 6 (R14), contender B6's half-1 generator: RL6 for 6,000 updates (as many as B6's deployed checkpoint) on half 1's RL scenes, 0 scorer errors; navdev single path 83.72 (the deployed B6 86.74) | 40,943 (half 1) | 500-6,000 (12) | 2.4 | rl_6000 `9b9289bf634e04e2` | yes | yes |
| `selector-c4-gen-half0-rl` | `selector-c4-gen-half0` epoch 10 | phase 6 (R14), contender C4's half-0 generator: RL4 for 2,500 updates (as many as C4's deployed checkpoint) on half 0's RL scenes, 0 scorer errors; navdev single path 82.68 (the deployed C4 87.28) | 39,683 (half 0) | 500-2,500 (5) | 1.3 | rl_2500 `edd4ebaed16e6439` | yes | yes |
| `selector-c4-gen-half1-rl` | `selector-c4-gen-half1` epoch 10 | phase 6 (R14), contender C4's half-1 generator: RL4 for 2,500 updates (as many as C4's deployed checkpoint) on half 1's RL scenes, 0 scorer errors; navdev single path 79.42 (the deployed C4 87.28) | 40,943 (half 1) | 500-2,500 (5) | 1.3 | rl_2500 `f692019589017283` | yes | yes |
| `selector-c6-gen-half0-rl` | `selector-c6-gen-half0` epoch 10 | phase 6 (R14), contender C6's half-0 generator: RL6 for 4,500 updates (as many as C6's deployed checkpoint) on half 0's RL scenes, 0 scorer errors; navdev single path 86.06 (the deployed C6 87.43) | 39,683 (half 0) | 500-4,500 (9) | 1.9 | rl_4500 `1fab7f20932db0cd` | yes | yes |
| `selector-c6-gen-half1-rl` | `selector-c6-gen-half1` epoch 10 | phase 6 (R14), contender C6's half-1 generator: RL6 for 4,500 updates (as many as C6's deployed checkpoint) on half 1's RL scenes, 0 scorer errors; navdev single path 85.24 (the deployed C6 87.43) | 40,943 (half 1) | 500-4,500 (9) | 1.9 | rl_4500 `11414a539c690719` | yes | yes |
| `selector-d6-gen-half0-rl` | `selector-d6-gen-half0` epoch 5 | phase 6 (R14), contender D6's half-0 generator: RL6 for 3,000 updates (as many as D6's deployed checkpoint) on half 0's RL scenes; navdev single path 79.97 (the deployed D6 82.65) | 39,683 (half 0) | 500-3,000 (6) | 1.5 | rl_3000 `9098dcd5d7514a1f` | yes | yes |
| `selector-d6-gen-half1-rl` | `selector-d6-gen-half1` epoch 5 | phase 6 (R14), contender D6's half-1 generator: RL6 for 3,000 updates (as many as D6's deployed checkpoint) on half 1's RL scenes, 0 scorer errors; navdev single path 77.92 (the deployed D6 82.65) | 40,943 (half 1) | 500-3,000 (6) | 1.5 | rl_3000 `96322ea6196a033e` | yes | yes |
| `b-r153-l12-18-24-rlfull4` | `b-r153-l12-18-24` epoch 10 | phase 6 (R16, R15, R18), full RL stage, RL4; round 1 of 5,000 updates; navdev choice the start (update 0, single path 79.34; every save 40.0-63.3), so R19 gives B's phase-1 configuration the same stage | 80,626 | 500-5,000 (10) | 2.0 | rl_500 `fb4e7f026d67487e`, rl_5000 `efbfa13a711f36fd` | yes | yes |
| `b-l12-18-24-rlfull4` | `b-l12-18-24` epoch 10 | phase 6 (R16, R18, R19), full RL stage, RL4, of B's phase-1 configuration (R19: the tuned one chose update 0); round 1 of 5,000 updates; navdev choice rl_1000 (85.30), above the tuned start's 79.34, so B4 deploys it | 80,626 | 500-5,000 (10) | 2.0 | rl_1000 `ee98fd231f3e3659`, rl_5000 `41f94a75e604f8b9` | yes | yes |
| `b-r153-l12-18-24-rlfull6` | `b-r153-l12-18-24` epoch 10 | phase 6 (R16, R15, R18), full RL stage, RL6; 2 rounds (10,000 updates: round 1's last save was navdev's best, round 2's was not); navdev choice rl_6000 (86.74), so B6 deploys it | 80,626 | 500-10,000 (20) | 3.6 | rl_6000 `193fbe952a13accc`, rl_10000 `9eca39c2ec345619` | yes | yes |
| `c-r152-l12-18-24-rlfull4` | `c-r152-l12-18-24` epoch 10 | phase 6 (R16, R15, R18), full RL stage, RL4; round 1 of 5,000 updates; navdev choice rl_2500 (87.28, on a plateau of 86.2-87.3 from update 1,500), so C4 deploys it and no round 2 | 80,626 | 500-5,000 (10) | 2.0 | rl_2500 `452bb042a575adc5`, rl_5000 `d85ca44c9a935596` | yes | yes |
| `c-r152-l12-18-24-rlfull6` | `c-r152-l12-18-24` epoch 10 | phase 6 (R16, R15, R18), full RL stage, RL6; round 1 of 5,000 updates; navdev choice rl_4500 (87.43), not the last save, so no round 2 | 80,626 | 500-5,000 (10) | 2.0 | rl_4500 `0d0f3afbcb0b267e`, rl_5000 `8d91138fde9b11fa` | yes | yes |
| `d-r153-l12-18-24-rlfull4` | `d-r153-l12-18-24` epoch from R11 via R15 | phase 6 (R16, R15, R18), full RL stage, RL4; round 1 of 5,000 updates; navdev choice the start (update 0, 76.57; every save 64.9-75.2), so R19 gives D's phase-1 configuration the same stage | 80,626 | 500-5,000 (10) | 2.0 | rl_500 `3eb835caabdff946`, rl_5000 `1172f2b327c2f029` | yes | yes |
| `d-l12-18-24-rlfull4` | `d-l12-18-24` epoch 5 | phase 6 (R16, R18, R19), full RL stage, RL4, of D's phase-1 configuration (R19: the tuned one kept its start); round 1 of 5,000 updates; navdev choice the start (77.10; saves fell from 74.6 to 20.8), so D4 deploys `d-l12-18-24` epoch 5 without RL (77.10 above the tuned start's 76.57) | 80,626 | 500-5,000 (10) | 2.0 | rl_500 `7aded266b51a3cf8`, rl_5000 `117711caf0d6cbc1` | yes | yes |
| `d-r153-l12-18-24-rlfull6` | `d-r153-l12-18-24` epoch from R11 via R15 | phase 6 (R16, R15, R18), full RL stage, RL6; round 1 of 5,000 updates; navdev choice rl_3000 (82.65, from the start's 76.57), not the last save, so no round 2 | 80,626 | 500-5,000 (10) | 2.0 | rl_3000 `9c0dcbdadb38b287`, rl_5000 `413237c11dbb0e1c` | yes | yes |

A's RL7 is the same run as its RL0 (A's steps are not Gaussian, so the SD floor does nothing). All R12 and R13 recipe runs are on both machines with matching checksums (2026-11-20 10:49).

The RL trial lowered navdev's single-path PDMS (81.18 → 73.97 after 200 updates), which led to R12 in `docs/HEADS_PLAN.md`. It is kept as the evidence for that rule, not used further.

### Selectors

A selector run's folder (`exp/heads/<run>/`) holds `epoch_1.pt` … `epoch_20.pt`, `checkpoint.pt` (with the optimizer), `run.json`, `log.jsonl`, `choice.json` (the epoch and own-pick margin chosen on the 10 held-out training drives, R14) and `navdev/` (its selected paths and R14's measures). Each is trained on its chain's out-of-fold candidates for the RL scenes (70,889 scenes, 7 candidates each; 9,737 scenes of 10 drives held out). The SHA-256 is of the chosen epoch's weights; all 21 `.pt` files match on both machines (checked 2026-11-24 04:59).

| Run | Chain (deployed path generator) | Variant | Chosen on held-out drives | Parameters | Training time | GB | SHA-256 (first 16) | Desktop | Anvil |
|---|---|---|---|---|---|---|---|---|---|
| `selector-rehearsal-xfit-s0` | the rehearsal, no contender: `a1-l12-18-24` epoch 20, without RL | cross-fitted, seed 0; navdev +1.35 over the own pick (used for no choice) | epoch 10, margin 0.02 | 7,199,493 | 34.3 min | 0.6 | epoch_10 `baf803adb6bee0ff` | yes | yes |
| `selector-rehearsal-xfit-boundary` | the same | cross-fitted, seed 0, with 4 boundary paths per scene in training only (an ablation); navdev +2.08 | epoch 12, margin 0.03 | 7,199,493 | 41.7 min | 0.6 | epoch_12 `3a195a943d1f66b7` | yes | yes |
| `selector-a4-xfit-s0` | contender A4: `a3-l12-18-24-rlfull4` rl_2500 | cross-fitted, seed 0, the one that decides adoption (R14); navdev -0.36, not adopted | epoch 5, margin 0.03 | 7,199,493 | 36.9 min | 0.6 | epoch_5 `33d6a9d0bea14da4` | yes | yes |
| `selector-a4-xfit-s1` | contender A4: `a3-l12-18-24-rlfull4` rl_2500 | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev -0.37 (-1.41 to +0.55), as seed 0's -0.36 | epoch 10, margin 0.01 | 7,199,493 | 37.6 min | 0.6 | epoch_10 `61fc7b148a2ffacf` | yes | yes |
| `selector-a4-xfit-pdms` | contender A4: `a3-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev -0.55 (-1.38 to +0.23) | epoch 20, margin 0.02 | 7,198,465 | 36.8 min | 0.6 | epoch_20 `9b02b04d1ce612ca` | yes | yes |
| `selector-a4-xfit-listwise` | contender A4: `a3-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev 0.00 (-0.16 to +0.08), switching in 7.7% of scenes | epoch 20, margin 0.1 | 7,199,493 | 39.8 min | 0.6 | epoch_20 `74bc93d0248a2348` | yes | yes |
| `selector-a4-infold-s0` | contender A4: `a3-l12-18-24-rlfull4` rl_2500 | in fold, seed 0, R14's control (trained on the deployed A4's own candidates; reported, decides nothing); navdev -0.21 (-0.70 to -0.01) | epoch 9, margin 0.05 | 7,199,493 | 40.8 min | 0.6 | epoch_9 `a7d2fca3d74a9b4f` | yes | yes |
| `selector-a4-xfit-imitation` | contender A4: `a3-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev -1.55 (-2.46 to -0.35) | epoch 10, margin 0.01 | 7,199,750 | 42.9 min | 0.6 | epoch_10 `f91aa86a34e8e8cf` | yes | yes |
| `selector-a4-xfit-boundary` | contender A4: `a3-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev -0.38 (-1.92 to +1.05) | epoch 7, margin 0.02 | 7,199,493 | 43.0 min | 0.6 | epoch_7 `179f2a8792535011` | yes | yes |
| `selector-a6-xfit-s0` | contender A6: `a3-l12-18-24-rlfull6` rl_3000 | cross-fitted, seed 0, the one that decides adoption (R14); navdev +0.13, adopted (no gain claimed) | epoch 12, margin 0.1 | 7,199,493 | 41.7 min | 0.6 | epoch_12 `9d7138f989cd6230` | yes | yes |
| `selector-a6-xfit-s1` | contender A6: `a3-l12-18-24-rlfull6` rl_3000 | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev +0.27 (0.00 to +0.52), seed 0 +0.13 | epoch 6, margin 0.1 | 7,199,493 | 40.3 min | 0.6 | epoch_6 `6a7e9bd9427feece` | yes | yes |
| `selector-a6-xfit-pdms` | contender A6: `a3-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev -0.05 (-0.10 to +0.02) | epoch 9, margin 0.1 | 7,198,465 | 44.8 min | 0.6 | epoch_9 `84f0767d86f64bd6` | yes | yes |
| `selector-a6-xfit-listwise` | contender A6: `a3-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev +0.22 (0.00 to +0.33), switching in 0.7% of scenes | epoch 8, margin 0.1 | 7,199,493 | 36.5 min | 0.6 | epoch_8 `19560f4eb61cec7e` | yes | yes |
| `selector-a6-xfit-imitation` | contender A6: `a3-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev +0.13 (0.00 to +0.25) | epoch 8, margin 0.1 | 7,199,750 | 43.0 min | 0.6 | epoch_8 `228d87a8f21c0c03` | yes | yes |
| `selector-a6-infold-s0` | contender A6: `a3-l12-18-24-rlfull6` rl_3000 | in fold, seed 0, R14's control (trained on the deployed A6's own candidates; reported, decides nothing); navdev +0.14 (-0.34 to +0.61) | epoch 18, margin 0.1 | 7,199,493 | 35.9 min | 0.6 | epoch_18 `8235d4f3d0c8d872` | yes | yes |
| `selector-a6-xfit-boundary` | contender A6: `a3-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev +0.14 (-0.49 to +1.01), switching in 26.9% of scenes | epoch 11, margin 0.05 | 7,199,493 | 41.5 min | 0.6 | epoch_11 `bebacda04bd1407b` | yes | yes |
| `selector-b4-xfit-s0` | contender B4: `b-l12-18-24-rlfull4` rl_1000 (R19) | cross-fitted, seed 0, the one that decides adoption (R14); navdev -0.43, not adopted | epoch 17, margin 0.05 | 7,199,493 | 40.3 min | 0.6 | epoch_17 `44f98ad6787188d9` | yes | yes |
| `selector-b4-xfit-s1` | contender B4: `b-l12-18-24-rlfull4` rl_1000 | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev +0.06 (-0.50 to +0.31), seed 0 -0.43 | epoch 5, margin 0.05 | 7,199,493 | 41.5 min | 0.6 | epoch_5 `2343bfec6988920d` | yes | yes |
| `selector-b4-xfit-pdms` | contender B4: `b-l12-18-24-rlfull4` rl_1000 | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev -0.17 (-1.62 to +1.05) | epoch 20, margin 0.02 | 7,198,465 | 41.9 min | 0.6 | epoch_20 `b378ce2ce600a48a` | yes | yes |
| `selector-b4-xfit-listwise` | contender B4: `b-l12-18-24-rlfull4` rl_1000 | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev +0.12 (-0.42 to +0.34) | epoch 9, margin 0.1 | 7,199,493 | 37.0 min | 0.6 | epoch_9 `31a8105e58e98235` | yes | yes |
| `selector-b4-infold-s0` | contender B4: `b-l12-18-24-rlfull4` rl_1000 | in fold, seed 0, R14's control (trained on the deployed B4's own candidates; reported, decides nothing); navdev +0.48 (-0.17 to +0.90) | epoch 13, margin 0.05 | 7,199,493 | 43.9 min | 0.6 | epoch_13 `169f8f8a9d3d82ee` | yes | yes |
| `selector-b4-xfit-imitation` | contender B4: `b-l12-18-24-rlfull4` rl_1000 | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev -0.18 (-0.91 to +0.35) | epoch 11, margin 0.02 | 7,199,750 | 43.3 min | 0.6 | epoch_11 `d5f13885d8739bd5` | yes | yes |
| `selector-b4-xfit-boundary` | contender B4: `b-l12-18-24-rlfull4` rl_1000 | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev +0.07 (-0.55 to +0.65), switching in 26.8% of scenes | epoch 6, margin 0.1 | 7,199,493 | 44.5 min | 0.6 | epoch_6 `3848083b06b49011` | yes | yes |
| `selector-b6-xfit-s0` | contender B6: `b-r153-l12-18-24-rlfull6` rl_6000 | cross-fitted, seed 0, the one that decides adoption (R14); navdev -1.87, not adopted | epoch 10, margin 0.03 | 7,199,493 | 39.4 min | 0.6 | epoch_10 `29cb5b79fff70712` | yes | yes |
| `selector-b6-xfit-s1` | contender B6: `b-r153-l12-18-24-rlfull6` rl_6000 | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev -0.64 (-1.35 to -0.20), seed 0 -1.87 | epoch 14, margin 0.1 | 7,199,493 | 52.4 min | 0.6 | epoch_14 `bd361f8b5b166ce5` | yes | yes |
| `selector-b6-xfit-pdms` | contender B6: `b-r153-l12-18-24-rlfull6` rl_6000 | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev -0.30 (-1.09 to +0.90) | epoch 5, margin 0.05 | 7,198,465 | 41.8 min | 0.6 | epoch_5 `157e54f859bae095` | yes | yes |
| `selector-b6-xfit-listwise` | contender B6: `b-r153-l12-18-24-rlfull6` rl_6000 | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev +0.01 (-0.25 to +0.50) | epoch 17, margin 0.1 | 7,199,493 | 35.3 min | 0.6 | epoch_17 `74947992950c1e3c` | yes | yes |
| `selector-b6-infold-s0` | contender B6: `b-r153-l12-18-24-rlfull6` rl_6000 | in fold, seed 0, R14's control (trained on the deployed B6's own candidates; reported, decides nothing); navdev -1.09 (-1.70 to -0.62) | epoch 14, margin 0.02 | 7,199,493 | 39.2 min | 0.6 | epoch_14 `1f53111ab7e0e36f` | yes | yes |
| `selector-b6-xfit-imitation` | contender B6: `b-r153-l12-18-24-rlfull6` rl_6000 | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev -1.10 (-1.56 to -0.19) | epoch 5, margin 0.03 | 7,199,750 | 43.1 min | 0.6 | epoch_5 `a9d10bef198c4088` | yes | yes |
| `selector-b6-xfit-boundary` | contender B6: `b-r153-l12-18-24-rlfull6` rl_6000 | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev -1.61 (-2.77 to -0.61), switching in 37.8% of scenes | epoch 11, margin 0.02 | 7,199,493 | 37.0 min | 0.6 | epoch_11 `db935428738bf93d` | yes | yes |
| `selector-c4-xfit-s0` | contender C4: `c-r152-l12-18-24-rlfull4` rl_2500 | cross-fitted, seed 0, the one that decides adoption (R14); navdev -0.40, not adopted | epoch 7, margin 0.03 | 7,199,493 | 36.7 min | 0.6 | epoch_7 `bd04357b58436379` | yes | yes |
| `selector-c4-xfit-s1` | contender C4: `c-r152-l12-18-24-rlfull4` rl_2500 | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev -0.47 (-0.87 to -0.19), seed 0 -0.40 | epoch 14, margin 0.05 | 7,199,493 | 43.7 min | 0.6 | epoch_14 `a992b0bc5f74a96e` | yes | yes |
| `selector-c4-xfit-pdms` | contender C4: `c-r152-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev -0.66 (-1.21 to +0.19) | epoch 6, margin 0.03 | 7,198,465 | 40.6 min | 0.6 | epoch_6 `45f3b4bd0c46a917` | yes | yes |
| `selector-c4-xfit-listwise` | contender C4: `c-r152-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev -0.36 (-0.64 to +0.03) | epoch 13, margin 0.05 | 7,199,493 | 36.8 min | 0.6 | epoch_13 `6e55b4839f1c20b9` | yes | yes |
| `selector-c4-infold-s0` | contender C4: `c-r152-l12-18-24-rlfull4` rl_2500 | in fold, seed 0, R14's control (trained on the deployed C4's own candidates; reported, decides nothing); navdev -0.01 (-0.03 to 0.00), switching in 2.6% of scenes | epoch 7, margin 0.1 | 7,199,493 | 48.1 min | 0.6 | epoch_7 `c43db22a2543db44` | yes | yes |
| `selector-c4-xfit-imitation` | contender C4: `c-r152-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev -0.40 (-0.95 to +0.18) | epoch 10, margin 0.05 | 7,199,750 | 53.2 min | 0.6 | epoch_10 `4f540f2f6c14a0f9` | yes | yes |
| `selector-c4-xfit-boundary` | contender C4: `c-r152-l12-18-24-rlfull4` rl_2500 | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev -1.54 (-2.47 to -0.74), switching in 48.5% of scenes | epoch 13, margin 0.03 | 7,199,493 | 38.7 min | 0.6 | epoch_13 `d22b3704339b90b9` | yes | yes |
| `selector-c6-xfit-s0` | contender C6: `c-r152-l12-18-24-rlfull6` rl_4500 | cross-fitted, seed 0, the one that decides adoption (R14); navdev -0.26, not adopted | epoch 7, margin 0.1 | 7,199,493 | 37.0 min | 0.6 | epoch_7 `c564b533e34de646` | yes | yes |
| `selector-c6-xfit-s1` | contender C6: `c-r152-l12-18-24-rlfull6` rl_4500 | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev -0.18 (-0.49 to 0.00), seed 0 -0.26 | epoch 3, margin 0.1 | 7,199,493 | 42.0 min | 0.6 | epoch_3 `7bc28d5134480035` | yes | yes |
| `selector-c6-xfit-pdms` | contender C6: `c-r152-l12-18-24-rlfull6` rl_4500 | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev 0.00, switching in 0.1% of scenes | epoch 3, margin 0.1 | 7,198,465 | 41.2 min | 0.6 | epoch_3 `859d4fdade0b01eb` | yes | yes |
| `selector-c6-xfit-listwise` | contender C6: `c-r152-l12-18-24-rlfull6` rl_4500 | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev 0.00, switching in 0.3% of scenes | epoch 5, margin 0.1 | 7,199,493 | 39.9 min | 0.6 | epoch_5 `3a619dde8e1af243` | yes | yes |
| `selector-c6-infold-s0` | contender C6: `c-r152-l12-18-24-rlfull6` rl_4500 | in fold, seed 0, R14's control (trained on the deployed C6's own candidates; reported, decides nothing); navdev +0.01 (0.00 to +0.01) | epoch 1, margin 0.1 | 7,199,493 | 40.0 min | 0.6 | epoch_1 `59e75dba20c89226` | yes | yes |
| `selector-c6-xfit-imitation` | contender C6: `c-r152-l12-18-24-rlfull6` rl_4500 | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev 0.00, never switching | epoch 4, margin 0.1 | 7,199,750 | 38.5 min | 0.6 | epoch_4 `d3d999888748efcf` | yes | yes |
| `selector-c6-xfit-boundary` | contender C6: `c-r152-l12-18-24-rlfull6` rl_4500 | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev -0.79 (-1.17 to +0.03) | epoch 8, margin 0.1 | 7,199,493 | 38.1 min | 0.6 | epoch_8 `d31ebdc58067b26b` | yes | yes |
| `selector-d4-xfit-s0` | contender D4: `d-l12-18-24` epoch_5 (no RL) | cross-fitted, seed 0, the one that decides adoption (R14); navdev +6.02 (+2.70 to +7.78), adopted, gain claimed | epoch 10, margin 0.0 | 7,199,493 | 45.5 min | 0.6 | epoch_10 `5339364ceaabf00e` | yes | yes |
| `selector-d4-xfit-s1` | contender D4: `d-l12-18-24` epoch_5 (no RL) | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev +5.90 (+3.51 to +6.89), seed 0 +6.02 | epoch 9, margin 0.005 | 7,199,493 | 44.1 min | 0.6 | epoch_9 `bb4f36956102cb79` | yes | yes |
| `selector-d4-xfit-pdms` | contender D4: `d-l12-18-24` epoch_5 (no RL) | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev +5.28 (+2.85 to +6.41) | epoch 10, margin 0.01 | 7,198,465 | 48.0 min | 0.6 | epoch_10 `ecd7de0b149f015e` | yes | yes |
| `selector-d4-xfit-listwise` | contender D4: `d-l12-18-24` epoch_5 (no RL) | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev +4.77 (+2.03 to +6.14) | epoch 10, margin 0.02 | 7,199,493 | 39.1 min | 0.6 | epoch_10 `edf81c13e51e2315` | yes | yes |
| `selector-d4-infold-s0` | contender D4: `d-l12-18-24` epoch_5 (no RL) | in fold, seed 0, R14's control (trained on the deployed D4's own candidates; reported, decides nothing); navdev +5.41 (+3.50 to +6.92) | epoch 9, margin 0.01 | 7,199,493 | 48.9 min | 0.6 | epoch_9 `626f02e053eb8807` | yes | yes |
| `selector-d4-xfit-imitation` | contender D4: `d-l12-18-24` epoch_5 (no RL) | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev +6.62 (+3.66 to +8.30) | epoch 10, margin 0.0 | 7,199,750 | 38.8 min | 0.6 | epoch_10 `f65fd78d73517761` | yes | yes |
| `selector-d4-xfit-boundary` | contender D4: `d-l12-18-24` epoch_5 (no RL) | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev +4.75 (+2.50 to +6.18) | epoch 10, margin 0.02 | 7,199,493 | 42.1 min | 0.6 | epoch_10 `80e2f080ac1b946d` | yes | yes |
| `selector-d6-xfit-s0` | contender D6: `d-r153-l12-18-24-rlfull6` rl_3000 | cross-fitted, seed 0, the one that decides adoption (R14); navdev -2.16 (-3.95 to -1.32), not adopted | epoch 8, margin 0.02 | 7,199,493 | 42.5 min | 0.6 | epoch_8 `5be357e5242e9979` | yes | yes |
| `selector-d6-xfit-s1` | contender D6: `d-r153-l12-18-24-rlfull6` rl_3000 | cross-fitted, seed 1 (R14's second seed; reported, decides nothing); navdev -2.62 (-5.44 to -0.97), seed 0 -2.16 | epoch 6, margin 0.02 | 7,199,493 | 47.7 min | 0.6 | epoch_6 `1d2285ed22ac3c9b` | yes | yes |
| `selector-d6-xfit-pdms` | contender D6: `d-r153-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's ablation predicting PDMS alone (reported, decides nothing); navdev -1.57 (-2.80 to +0.08) | epoch 6, margin 0.03 | 7,198,465 | 46.7 min | 0.6 | epoch_6 `af3ab5b948e787e5` | yes | yes |
| `selector-d6-xfit-listwise` | contender D6: `d-r153-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's listwise-loss ablation (reported, decides nothing); navdev -3.18 (-4.54 to -2.39), switching in 63.8% of scenes | epoch 6, margin 0.01 | 7,199,493 | 47.3 min | 0.6 | epoch_6 `61380c91480feed3` | yes | yes |
| `selector-d6-infold-s0` | contender D6: `d-r153-l12-18-24-rlfull6` rl_3000 | in fold, seed 0, R14's control (trained on the deployed D6's own candidates; reported, decides nothing); navdev +1.34 (-0.25 to +2.26) | epoch 5, margin 0.02 | 7,199,493 | 41.3 min | 0.6 | epoch_5 `1ac6a18374641967` | yes | yes |
| `selector-d6-xfit-imitation` | contender D6: `d-r153-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's imitation-target ablation (reported, decides nothing); navdev -1.18 (-3.01 to +0.12) | epoch 10, margin 0 | 7,199,750 | 39.4 min | 0.6 | epoch_10 `d6e21059e71879a3` | yes | yes |
| `selector-d6-xfit-boundary` | contender D6: `d-r153-l12-18-24-rlfull6` rl_3000 | cross-fitted, R14's ablation with the boundary candidates (reported, decides nothing); navdev -2.32 (-3.54 to -0.24), switching in 50.6% of scenes | epoch 9, margin 0.02 | 7,199,493 | 38.7 min | 0.6 | epoch_9 `b087d4918dc6afa3` | yes | yes |
