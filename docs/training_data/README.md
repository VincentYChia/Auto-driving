# Workstream B: training data. Design before build

**Status (2026-10-18): design phase.**
- `data_pipeline/` is a **prototype (v0.1)**. It was built to explore a format, has only been run on synthetic logs, and must **not** be used to produce data for experiments.
- Prototype format: `PROTOTYPE_v0.1_SCHEMA.md`.
- This workstream is independent of NAVSIM scoring (`docs/navsim_eval/`). Both share only the read-only NAVSIM access layer, `navsim_tools/`.

**Why slow down here:**
- Months of experiments will be trained and model-selected on this data: pilot and reasoner fine-tuning, the C1 and C2 comparisons, dev-set scores.
- A wrong frame convention, a leaky split or a bad label set would silently invalidate all of them.
- Reasoning labels are expensive to regenerate.
- So: agree on requirements, check the data itself, decide the design explicitly, then freeze versioned releases.

Status markers follow `docs/RESEARCH_DIRECTION.md`: [agreed] / [proposed] / [open].

---

## 1. What the data must support [proposed; confirm in Phase 1]

| # | Consumer | Needs |
|---|---|---|
| a | **Pilot SFT** (Qwen3.5, ~2B) | Current front image (± other cameras); ego state, history and route command; object-motion context; optional decision token. Target: 8 future poses at 0.5 s |
| b | **Reasoner SFT** (Gemma 4 E2B/E4B) | Image(s), ego state, object motion. Target: narration, reasoning, decision |
| c | **C1 ablations** | The *same* samples rendered with different temporal context: image only; ground-truth motion; tracker motion; multi-frame images. Matched token budgets |
| d | **C2 tokenizer study** | All training trajectories (continuous) for fitting codebooks and measuring rounding error vs score |
| e | **Motion encoder / tracker work** | Ground-truth tracks over history; paths to past sensor frames |
| f | **Model selection without navtest** | A frozen dev split from navtrain, scorable with NAVSIM's PDM score |
| g | **MVLAD-AD comparability** (optional) | nuScenes (+ Nu-X reasoning text) in the same format |
| h | **Edge budget** | Token count of every input view |

## 2. Principles [proposed]
1. **Layers:**
   - **raw:** NAVSIM logs, never modified;
   - **canonical:** lossless, versioned, validated;
   - **views:** regenerable; they carry research choices such as prompts, filters, tokenization and windows;
   - **labels:** versioned by model + prompt.
2. **The canonical layer is lossless.** Range filters, rounding, history length and prompt formats are view choices, not storage choices.
3. **Test-time-visible information is separated physically** from targets and privileged information (other agents' futures, map, route). A loader then cannot leak it by accident.
4. **Splits are by log and frozen.** NAVSIM test splits are never used for training or selection.
5. **Every convention is checked on real data**, with statistics *and* visual checks, before anything is frozen.
6. **Releases are immutable:** version, manifest with checksums, data card. Any change means a new version, and every experiment records the dataset version it used.

## 3. Why the v0.1 prototype is not the foundation
| # | Limitation | Consequence |
|---|---|---|
| L1 | Each record is a self-contained NAVSIM window (4 history + 10 future). navtrain uses `frame_interval: 1` (V, `scene_filter/navtrain.yaml`), so neighbouring records share most frames | Each frame is stored up to 14×, and the window shape is baked in. Other history/horizon lengths need a re-extraction |
| L2 | Lossy at the canonical layer: 100 m range filter, rounding, only the 4 frames NAVSIM shows at test time | Some research questions (longer history, far objects) need a re-extraction |
| L3 | No train/dev split | A random split would leak (overlapping windows, same-log correlation); model selection would fall back to navtest |
| L4 | Conventions checked only against NAVSIM's *code* and synthetic logs, not real data: box/velocity frames, track persistence, timing regularity, sensor sync, origin (rear axle vs LiDAR) | An undetected convention error would corrupt every motion feature |
| L5 | Privileged data lives in the same record as inputs, separated only by a key name | A loader bug could leak the future into training inputs |
| L6 | Labels uncalibrated: meta-action thresholds hand-picked, class balance unknown, teacher prompt untried. Hindsight labels may teach the reasoner to "know" the future | Label noise or bias would enter both pilot and reasoner training |
| L7 | No QA tooling: no BEV or camera-overlay visualisation, no real-data checks | Errors would only show up as unexplained training results |
| L8 | No release discipline: no per-file checksums, data card or licence review | Results can't be traced reliably to the exact data used |

Worth keeping from the prototype (tested, and likely to carry into v1):
- the SE(2) geometry and raw-log readers in `navsim_tools/`, cross-checked against the devkit;
- the privileged/test-time distinction;
- the view idea, including C1 variants on identical samples;
- provenance stamping.

## 4. Plan: phases with review points [proposed]

| Phase | Output | Exit criterion (your review) |
|---|---|---|
| **P1 Requirements** | Table of consumers (a–h + yours): inputs, targets, horizons, sensors, scale, priority for the next 3 months | You agree the table |
| **P2 Real-data audit** | Script + report on a small real subset (see below) | You've seen the plots and agree on conventions; issue list written |
| **P3 Design v1** | Design doc: decisions D1–D11 below, storage layout, validation checks, versioning | Signed off |
| **P4 Build + validate v1** | Extraction + validators. Run on a dev subset, then a visual spot-check of ~50 random samples, then the full run | Release v1.0 frozen: manifest, checksums, data card |
| **P5 Labels v1** | Meta-action rules calibrated on real distributions; teacher pilot (~200 samples, 2 teachers and/or 2 prompt styles) with rubric-based human rating | Choice of teacher and prompt, then labels v1 released separately |

**P2 audit contents** (on ~10–20 navtrain logs; images only for the visual checks):
- **Coverage:** fields, types and missing values; timestamp spacing; overlap between windows.
- **Conventions:**
  - project ground-truth boxes into CAM_F0 with the stored calibration;
  - bird's-eye-view plots of tracks over the history after ego-motion compensation (static objects must stay put);
  - provided velocity vs finite differences of positions (nuPlan's velocities are network estimates that "may be inconsistent with future positions", V);
  - whether track IDs persist across frames.
- **Distributions:**
  - speed, route command and curvature;
  - object counts by class and range;
  - meta-action class frequencies under the current thresholds;
  - motion-text token counts vs image token counts.

## 5. Decisions for P3 [open]: options and my current recommendation
| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Storage granularity | (a) self-contained windows (v0.1); (b) per-log frame tables + a window index | **(b)**: each frame stored once; any window or horizon later; natural log-level splits |
| D2 | Splits | Log-level dev split from navtrain (size %, stratified by city/map?) | **Yes**, frozen ID lists. Also a dev metric cache so PDMS can be computed without navtest (feasibility to verify in P2) |
| D3 | What the canonical layer keeps | All objects, full precision, whole logs vs filtered | **Keep everything**; filter in views |
| D4 | Privileged separation | Key inside the record vs separate stores | **Separate stores**: observations / targets / privileged |
| D5 | Format | Parquet (columnar, partial reads) vs gzipped JSONL | **Parquet** for canonical, **JSONL** for training views. `pyarrow` is already in NAVSIM's requirements (V) |
| D6 | Sources in v1 | navtrain only; + raw nuPlan DBs for the same logs (20 Hz); + nuScenes / Nu-X; + reasoning or synthetic sets. Table: `DATASETS.md` | **navtrain first.** Next candidate: the raw nuPlan DBs for the navtrain logs (same scenes at 20 Hz, longer history, same licence). nuScenes / Nu-X only if consumer g ranks high |
| D7 | Sensors referenced | Front camera only vs all 8 cameras + LiDAR + calibration | **All** (paths are cheap); image choice is a view concern |
| D8 | Decision vocabulary | Rule-based meta-actions vs teacher-defined | **Rule-based**: deterministic and checkable; calibrate in P2 |
| D9 | Reasoning labels | Teacher model; hindsight (sees the future) vs causal prompting; review protocol; budget | **Decide from the P5 pilot**, not upfront. Include ReCogDrive's released navtrain QA as a comparison arm (`DATASETS.md` §1g) |
| D10 | Licences / terms | OpenScene / nuPlan dataset terms; teacher model output terms | **Verify before generating or sharing derived data.** First pass in `DATASETS.md` §3; the full licence texts are not yet read |
| D11 | Versioning | Release policy | Semantic versions, immutable releases; experiments log the dataset version |

## 6. What I need from you to start P1
1. **Process:** agree or change the principles (§2) and the phases (§4).
2. **Priorities:** rank consumers a–h for the next 1–3 months and add any missing.
3. **P2 data:** OK to download the navtrain logs on Anvil (`trainval-logs`, ~14 GB, one archive) plus one of the 32 `navtrain-current` camera chunks for the visual checks? I'll write the audit script only after P1 is agreed.

## Using the prototype meanwhile
Only for environment smoke tests on a few logs:
```bash
MAX_LOGS=5 sbatch --partition=gpu-debug --time=00:30:00 scripts/anvil/data_pipeline/extract_navtrain.sbatch
```
Don't run the full extraction or any teacher labelling until the v1 design is agreed.
