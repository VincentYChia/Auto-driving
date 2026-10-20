# Other open datasets: survey for D6, D9 and D10 (2026-10-20)

**Status: research input [proposed].** Nothing here is adopted. Adding any source needs the researcher's sign-off under `README.md` §4–5.

**Evidence:** V / R / D as in `docs/RESEARCH_DIRECTION.md`.
- Only github.com and raw.githubusercontent.com were reachable. arXiv, Hugging Face and the vendor sites were blocked.
- So paper, dataset-card and leaderboard facts are R at best.
- Results claimed in a README are the authors' own, so they are R.

## 0. What NAVSIM itself gives (facts that shaped this survey)
- **Rate and history:** OpenScene is nuPlan downsampled to 2 Hz (V). At test time an agent sees 4 frames, a 1.5 s span (V).
- **Splits:** navtest and navhard are drawn from the OpenScene `test` split (V).
- **Object boxes are auto-labels, not human annotation.** nuPlan says the boxes "are extracted from an Offline Perception system and tracked across time" (V, `nuplan_schema.md`).
  - Velocities are "estimated by a Neural Network and may be inconsistent with future positions" (V).
  - Traffic-light status is estimated "from the observed motion of agents" (V).
  - So "ground-truth" motion in the C1 ablations means offline-perception motion (D).
- **No language labels** of any kind (V).
- **No high-rate closed loop.** v2 pseudo-simulation runs at 2 Hz, with one 4 s plan per stage (V).

## 1. Reasoning text (reasoner, consumer b; decision tokens)
| Source | Scenes | Size | Text | Made by | Split / leakage | Terms |
|---|---|---|---|---|---|---|
| **ReCogDrive NAVSIM QA** | navtrain, CAM_F0 | 752k QA (V) | QA: scene description; influential objects + advice; behaviour narration; decision explanation; lights, signs, pedestrians; planning (V) | Qwen2.5-VL-72B (V). The decision is computed from the ground-truth future, then explained: **hindsight** (V, generator code) | navtrain only (V) | Academic use; original licences apply (V) |
| **nuReasoning** (Motional; public Sep 2026) | Own 20 s clips at 10 Hz: 8 cameras, LiDAR, HD map, ego, objects, traffic lights, route commands (V). Camera names differ from nuPlan's, so probably a different rig (D) | 17k train / 2k val / 1k private-test clips, 105 h (V) | Spatial (~1 Hz), Decision and Counterfactual (every 5 s) annotations, served as QA: mostly multiple choice, plus free text and trajectories (V) | Gemini, then human check (R) | Relation to nuPlan logs not stated (V): check before use | Separate non-commercial and commercial terms (V) |
| **NuPlanQA-1M** | nuPlan train/val cameras (V) | 977,395 train QA (V) | Perception, spatial and ego-centric QA (R) | GPT-4o (R) | No navtest overlap expected (D); verify by log name | No licence file (V) |
| **Impromptu-VLA** | 8 datasets, NAVSIM included (R) | 80k clips (R) | Scene description, lights, decision explanation, meta-action, trajectory (V) | Rules + Qwen2.5-VL-72B (V/R) | Its code also builds a **navtest** set (V): drop it | Repo CC BY-SA 4.0 (V); base-data terms still apply (D) |
| **Nu-X** (Hint-AD) | nuScenes keyframes | 28,130 train + 6,019 val (V, counted) | One-line narration + one-line reasoning, ~10 words each (V, counted) | Human (V), reworded by an LLM (R) | n/a | No licence file (V); nuScenes non-commercial terms (R) |
| **BDD-X** | BDD100K dashcam | 6,970 videos, 26k actions (V) | Description + "because" explanation (V) | Human (V) | n/a | BDD100K terms (R) |

**Labelling methods (code, not data):**
- **AutoVLA:** an open 4-step chain-of-thought prompt that gives the ground-truth action as a hint (V). Academic licence (V).
- **NVIDIA Chain-of-Causation auto-labeller:** Apache-2.0, with Qwen3.5 or GPT-5.5 backends (V). It is built on trajdata, which reads nuPlan (V). Adapting it to NAVSIM is untested (D).

**Not released:** AutoVLA's nuPlan chain of thought, AdaThinkDrive, and the AutoDrive-P³ / AUGUstus free-text traces (V).

**Also exist (QA-style, mostly non-commercial):** DriveLM-nuScenes (360k graph QA, CC BY-NC-SA, V), NuInstruct (91k QA, V), OmniDrive (NVIDIA non-commercial, V), LingoQA (Wayve non-commercial, V), WOMD-Reasoning (~3M QA on Waymo motion data, V).

## 2. Object motion: higher rate, longer history (C1, motion encoder e, refresh loop)
| Source | Content | Rate / history | Labels | Terms | Use (D) |
|---|---|---|---|---|---|
| **Raw nuPlan DBs** | **The same logs NAVSIM samples:** ego, tracked boxes, traffic lights, route (V) | 20 Hz, whole logs (V) | Offline-perception auto-labels (V) | nuPlan terms, the same as OpenScene (V) | Higher-rate, longer motion history for navtrain scenes; nuPlan closed loop |
| nuPlan sensor blobs | 8 cameras + merged LiDAR (V) | Images 10 Hz, LiDAR 20 Hz (V); ~120–128 h (R); >20 TB (V) | As above | As above | Tracker training at 10 Hz on NAVSIM's own rig |
| Argoverse 2 Motion Forecasting | 250k × 11 s scenarios, 6 cities (V); no sensors | 10 Hz; 5 s history + 6 s future (R) | Tracks with inferred types; method not found (V) | CC BY-NC-SA 4.0 (V) | Longest standard history |
| Waymo Open Motion (WOMD) | 103,354 scenes (V); no raw images | 10 Hz; released windows have 1.1 s history + 8 s future (R) | Auto-labelled (R) | Waymo non-commercial (V) | Scale; history is short |
| Argoverse 2 Sensor | 1,000 logs × ~15 s (V) | Cameras 20 fps; cuboids with track IDs at 10 Hz (V) | Method not found | CC BY-NC-SA 4.0 (V) | Camera-paired tracks |
| Waymo Perception | 2,030 × 20 s segments (R) | 10 Hz (R) | Human (R) | Waymo non-commercial (V) | Camera-paired, human-labelled tracks |

## 3. Scale, diversity and the long tail (cameras + ego trajectory)
| Source | Size | Sensors | Labels | Terms |
|---|---|---|---|---|
| **NVIDIA PhysicalAI-AV** | 1,700+ h, 306k clips, 25 countries (V) | 7 cameras at ~30 fps, LiDAR, radar on some clips (V) | Egomotion ~100 Hz (V). Machine labels "coming soon"; reasoning labels "being added" (V) | NVIDIA AV Dataset Licence, gated (V). Limited to AV development "using NVIDIA technology" (R) |
| **WOD-E2E** | 4,021 runs (V), ≈12 h (R), rare long-tail events (R) | 8 cameras (R) | Ego past 4 s / future 5 s at 4 Hz; up to 3 rater-scored trajectories; no boxes (V) | Waymo non-commercial (V) |
| **L2D** (Yaak × Hugging Face) | Release R3: 100k episodes, 500+ h (V) | 6 cameras, GPS, IMU at 10 Hz (V) | CAN, OSM-matched route waypoints, instructions (V); no 3D objects (D) | Apache-2.0 (R) |
| ZOD | 1,473 × 20 s sequences, 14 countries (R) | 1 front camera (V), LiDAR (R) | Keyframe boxes (R) | CC BY-SA 4.0 (V) |
| CoVLA | 10k clips, 80+ h, Tokyo (R) | Front camera, CAN (R) | Automatic captions + trajectories (R) | Custom non-commercial; derivatives restricted (R) |

## 4. Closed loop: evaluating the refresh loop and stale reasoning (not training data)
| Simulator | Agent sees | Rate | Traffic | Notes | Terms |
|---|---|---|---|---|---|
| **nuPlan closed loop** | Tracked objects + 2 s buffer, ego, map, lights (V); cameras only log-replayed (D) | DB 20 Hz, sim 10 Hz, 15 s scenarios (V) | IDM vehicles; pedestrians open-loop (V) | NAVSIM's domain. Needs the full nuPlan DBs. No release since 2023 (V) | Devkit Apache-2.0; data non-commercial (V) |
| **CARLA + Bench2Drive** | 6 cameras, LiDAR, radar; privileged boxes (V) | 20 Hz evaluation (V) | Traffic Manager (V) | B2D-Robust injects 100–500 ms action latency (V). Short routes are near ceiling (LEAD 95.6 DS, R), so also report the long Town13 routes (D) | B2D data CC BY-NC-ND (V); PDM-Lite data CC BY 4.0 (V) |
| **AlpaSim** (NVIDIA) | Rendered cameras, ego, route; no agent tracks in its API (V) | 10 Hz decisions, asynchronous clocks (V) | Off by default (V) | Its nuPlan track evaluates locally on **navtest** scenes (V): evaluation only | Code Apache-2.0 (V) |
| Waymax / GPUDrive | Objects only (V) | 10 Hz (V) | Log replay, IDM or self-play (V) | Object-only RL | Waymax non-commercial; GPUDrive MIT (V) |

**Synthetic training data:**
- **SimScale:** 384K samples at 2 Hz (V), CC BY-NC-SA 4.0 (V). The authors report +5.1 to +8.6 navhard EPDMS (R). Its source split is not stated: check before use.
- **CARLA sets:**
  - LEAD: boxes at 20 Hz, ~1 TB (V); data licence not found.
  - PDM-Lite: CC BY 4.0 (V).
- **Cosmos-Drive-Dreams:** 81,802 synthetic videos; commercial use allowed (V).

**Poor fit, skipped:**
- Lyft L5: onboard-perception labels (V), one fixed route (R).
- ONCE: released at 2 Hz (V).
- PandaSet: small (R).
- HUGSIM: 4 Hz (V).
- NAVSIM v2 pseudo-sim, for the refresh loop: 2 Hz (V).

## 5. Licences and NAVSIM leakage
- **Non-commercial is the norm** (D: fine for papers; a product would need separate licences):
  - CC BY-NC-SA 4.0: nuPlan / OpenScene (V), Argoverse 2 (V), ONCE (V), nuScenes (R).
  - Waymo (V), nuReasoning's free tier (V), Bench2Drive (NC-ND, V).
- **ShareAlike** probably extends to derived data we publish, such as token sets and labels (D).
- **Permissive:** ZOD (CC BY-SA 4.0, V), PDM-Lite (CC BY 4.0, V), L2D (Apache-2.0, R), Cosmos-Drive-Dreams (V).
- **Restrictive:** PhysicalAI-AV (R). Read the full licence before relying on it.
- **Teacher outputs carry their own terms.** Qwen's licence asks for "Built with Qwen" on distributed models trained on its outputs (R). Gemini and GPT terms not checked.
- **NAVSIM rule:** never train on navtest / navhard / warmup / private splits (V). Concretely:
  - exclude OpenScene `test` logs when pulling raw nuPlan;
  - drop Impromptu-VLA's navtest set;
  - AlpaSim's nuPlan track and NavSafe-∞ are built on test logs (V): evaluation only;
  - check the source logs of SimScale, nuReasoning and NuPlanQA by log name.

## 6. Implications [proposed]
1. **D6:** navtrain stays the core of v1.
2. **D6, next source: the raw nuPlan DBs for the same logs.**
   - Same scenes, but at 20 Hz, with whole-log history, route and traffic lights. Same licence; no domain shift (D).
   - Serves C1 history length/rate studies, the motion encoder (e), the refresh loop and, later, the nuPlan closed loop.
   - Joining on log name + timestamp is to be verified in P2. DB size not yet checked.
3. **D9:** add ReCogDrive's navtrain QA as a comparison arm in the P5 teacher pilot. It is free, but model-written and hindsight. Borrow structure from the Chain-of-Causation and AutoVLA prompts.
4. **nuReasoning** is the most relevant new set (10 Hz, 8 cameras, decision and counterfactual reasoning). Before deciding, inspect:
   - its rig and cities;
   - overlap with nuPlan test logs;
   - how much of the label format is free text.
5. **nuScenes + Nu-X:** only if MVLAD-AD comparability (g) ranks high.
6. **Closed loop:** not a data decision yet. When the refresh loop exists:
   - nuPlan closed loop first: cheapest, same domain, object-level;
   - then CARLA + B2D-Robust for latency and staleness with images.
7. **Later, for pilot pretraining or external evaluation:** PhysicalAI-AV, WOD-E2E, L2D.

**Unverified:**
- size of the nuPlan DBs;
- source logs of nuReasoning, NuPlanQA and SimScale;
- nuReasoning's object-label method;
- how Argoverse 2 labels were made;
- ZOD sequence tracks;
- whether PhysicalAI-AV's machine labels and route data have shipped;
- the full licence texts.

## Sources (primary; read 2026-10-20)
- **nuPlan:**
  - `github.com/motional/nuplan-devkit`: `docs/nuplan_schema.md`, `docs/dataset_setup.md`, `nuplan/planning/scenario_builder/nuplan_db/nuplan_scenario.py`, `docs/metrics_description.md`.
  - **OpenScene:** `github.com/OpenDriveLab/OpenScene` (README, `docs/getting_started.md`).
  - **NAVSIM:** `github.com/autonomousvision/navsim` (`docs/splits.md`, `docs/metrics.md`).
- **Reasoning text:**
  - `github.com/xiaomi-research/recogdrive`: README, `navsim/planning/script/run_generate_dataset_pipeline.py`.
  - `github.com/nureasoning/nureasoning-devkit`: README, `docs/reasoning.md`.
  - `github.com/sungyeonparkk/NuPlanQA`.
  - `github.com/ahydchh/Impromptu-VLA`: `data_qa_generate/data_engine/datasets/navsim/dataset_navsim.py`.
  - `github.com/Robot-K/Hint-AD`: release v1.0.
  - `github.com/JinkyuKimUCB/BDD-X-dataset`.
  - `github.com/ucla-mobility/AutoVLA`: `dataset_utils/preprocessing/cot_prompts.py`.
  - `github.com/NVlabs/alpamayo-coc-autolabeler`.
  - `github.com/OpenDriveLab/DriveLM`, `github.com/xmed-lab/NuInstruct`, `github.com/NVlabs/OmniDrive`, `github.com/wayveai/LingoQA`, `github.com/yhli123/WOMD-Reasoning`.
- **Motion and sensors:**
  - `github.com/argoverse/user-guide` (datasets, terms).
  - `github.com/waymo-research/waymo-open-dataset` (LOG.md, protos).
  - `github.com/NVlabs/physical_ai_av` (wiki).
  - `github.com/huggingface/blog` (`lerobot-goes-to-driving-school.md`).
  - `github.com/zenseact/zod`.
- **Closed loop:**
  - `github.com/Thinklab-SJTU/Bench2Drive` and `Bench2Drive-Robust`.
  - `github.com/autonomousvision/carla_garage` (leaderboard_2).
  - `github.com/kesai-labs/lead`.
  - `github.com/NVlabs/alpasim` (`e2e_challenge/README.md`).
  - `github.com/waymo-research/waymax`, `github.com/Emerge-Lab/gpudrive`.
  - `github.com/OpenDriveLab/SimScale`.
  - `github.com/VAIL-UCLA/NavSafe-infty`.
  - `github.com/nv-tlabs/Cosmos-Drive-Dreams`.
