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
