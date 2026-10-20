# Driving datasets: master table (2026-10-20)

**Status: research input [proposed].** Nothing here is adopted. Adding any source needs the researcher's sign-off under `README.md` §4–5.

Ways to obtain object motion: `docs/OBJECT_MOTION.md`.

**Evidence:** V / R / D as in `docs/RESEARCH_DIRECTION.md`.
- Each cell's mark is its **weakest** evidence.
- Strengths and weaknesses are judgements for this project (D): a NAVSIM-scored, edge-deployed pilot plus a reasoner.
- Only github.com and raw.githubusercontent.com were reachable; arXiv, Hugging Face and most dataset sites were blocked. So paper and dataset-card facts are R at best, and every score is R.

## 0. What NAVSIM itself gives
- **Rate and history:** OpenScene is nuPlan downsampled to 2 Hz (V). At test time an agent sees 4 frames, a 1.5 s span (V).
- **Splits:** navtest and navhard are drawn from the OpenScene `test` split (V).
- **Object boxes are auto-labels, not human annotation.** nuPlan says the boxes "are extracted from an Offline Perception system and tracked across time" (V, `nuplan_schema.md`).
  - Velocities are "estimated by a Neural Network and may be inconsistent with future positions" (V).
  - Traffic-light status is estimated "from the observed motion of agents" (V).
  - So "ground-truth" motion in the C1 ablations means offline-perception motion (D).
- **No radar:** OpenScene lists 8 cameras and 5 LiDARs (V).
- **No LiDAR on navhard / v2 challenge test frames:** the maintainers said none would be provided for "any test frames (real or synthetic)" (V, NAVSIM issue #63, 2025).
- **Occupancy labels exist for trainval,** the source of navtrain (V, OpenScene v1.0). Flow labels are claimed but not verified (§1e).
- **No language labels** of any kind (V).
- **No high-rate closed loop.** v2 pseudo-simulation runs at 2 Hz, with one 4 s plan per stage (V).

## 1. Master table
Columns: **Dataset** · **Data type** · **Scale · rate** · **Labels: motion · text** · **Strengths** · **Weaknesses** · **Licence · leakage**.

### 1a. NAVSIM and its source data
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **NAVSIM navtrain** (OpenScene) | Sensor logs: 8 cameras, merged LiDAR, ego status, map | ~103k samples; 2 Hz; sensors 445 GB · V | Boxes + velocity + track IDs at 2 Hz, auto-labelled; no text · V | The evaluation domain; PDM scoring; ego status + route command | 2 Hz; 1.5 s of history at test time; network-estimated velocities; no text | CC BY-NC-SA 4.0 + nuPlan terms · V. Train on navtrain only |
| **Raw nuPlan DBs** | Object-level logs: ego, tracked boxes, traffic lights, route; no images | 15,000+ logs, ~1,300 h; 20 Hz; 1.22 TB for all splits · V | 20 Hz tracks + velocities, auto-labelled · V | The same logs as NAVSIM at 10× the rate, with whole-log history; needed for the nuPlan closed loop | No sensors; auto-labels; devkit last released 2023 | nuPlan terms, non-commercial · V. Exclude OpenScene `test` logs |
| **nuPlan sensor blobs** | 8 cameras + 5 LiDARs (merged) | ~120–128 h (R); cameras 10 Hz, LiDAR 20 Hz; 19.06 TB, readable per file (V) | Via the DB tracks · V | Full-rate vision on NAVSIM's own rig, for all 1,192 navtrain logs (V): the data for training a 10 Hz tracker | Very large; nuPlan account and terms required | As above. The test blobs hold every navtest / navhard log (V): never train on them |

### 1b. Sensor logs with 3D tracks (other rigs)
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **nuScenes** | 6 cameras 12 Hz, LiDAR 20 Hz, 5 radars 13 Hz (R); CAN bus, map (V) | 1,000 × 20 s ≈ 5.6 h · R | Human 3D boxes at 2 Hz keyframes (V); many text sets (§1g) | Human labels; radar; the most-reported benchmark; MVLAD-AD's training data | Small; labels only at 2 Hz; different rig and cities | CC BY-NC-SA 4.0 (R); commercial use by contact (V) |
| **Waymo Perception** | 5 cameras, 5 LiDARs | 2,030 × 20 s, ~390k frames, 10 Hz · R | Human 3D boxes with track IDs at 10 Hz · R | High-rate, human-labelled tracks; diverse US cities | No planning benchmark; different rig; v2.0 dropped the maps (V) | Waymo non-commercial · V |
| **Argoverse 2 Sensor** | 7 ring + 2 stereo cameras 20 fps, 2 LiDARs 10 Hz, map | 1,000 × ~15 s, 4.2 h, 1 TB · V | Cuboids with track IDs at 10 Hz, 26 classes · V | High-rate cameras + tracks; public S3, no account | Small; short logs; label method not found | CC BY-NC-SA 4.0 · V |
| **Argoverse 2 LiDAR** | LiDAR sequences | 20,000 × 30 s · V | Unlabelled · V | Self-supervised LiDAR or scene-flow pretraining | No labels | CC BY-NC-SA 4.0 · V |
| **nuReasoning** (Motional, Sep 2026) | 8 cameras, LiDAR, HD map, ego, traffic lights, route commands | 20k × 20 s = 105 h; 10 Hz · V | Object annotations at 10 Hz (method not found); spatial / decision / counterfactual QA · V | NAVSIM-like inputs at 10 Hz, plus reasoning and a planning benchmark | Very new; rig differs from nuPlan's (D); QA mostly multiple choice; relation to nuPlan logs unknown | Non-commercial + commercial terms · V. Check log overlap |
| **KITScenes** (Multimodal; LongTail) | Multimodal: 6 cameras + 128-beam LiDAR, Lanelet2 map. LongTail: 6 cameras at 5 Hz | ~1,000 scenes at 10 Hz; LongTail ~1,000 × 9 s · V | Not found; LongTail has instructions + counterfactual futures · V | German long-tail scenes with instructions | LongTail mostly unreleased (R) | Not found |
| **PandaSet** | 6 cameras, 2 LiDARs | 100+ × 8 s, 10 Hz (D) · R | Per-frame cuboids, 28 classes · V | Possibly commercial-friendly | Tiny; licence and download status unverified | "No-cost commercial" · R |
| **ONCE** | 7 cameras, 1 LiDAR | 144 h, 1M LiDAR frames; released at 2 Hz · V | 15–16k annotated scenes, 5 classes · V | Large unlabelled pretraining set | 2 Hz release; few labels | CC BY-NC-SA 4.0 + extra terms · V |
| **ZOD** | Front camera 8 MP (V); LiDARs 10 Hz, GNSS/IMU 100 Hz (R) | 100k frames, 1,473 × 20 s sequences, 14 countries · R | Keyframe 2D/3D boxes to 245 m (R); sequence tracks not found | Permissive; diverse Europe; long range | One camera | CC BY-SA 4.0 · V |
| **KITTI** (raw + tracking) | HDL-64 LiDAR, stereo grey + colour cameras, RTK GPS/IMU · V | Raw ≈ ¼ of 6 h recorded; tracking 21 train / 29 test sequences · R | Raw: 3D tracklets; tracking: per-frame boxes + track IDs, 8 classes, 10 Hz · R | Many baselines; tracks + GPS/IMU | Small; old sensors; front-camera labels only | CC BY-NC-SA 3.0 · V |
| **KITTI-360** | 2 perspective + 2 fisheye cameras (V); Velodyne, SICK, OXTS (R) | 320k images, 100k scans, 73.7 km · V | Dynamic boxes per timestamp with global IDs; dense semantic + instance labels · V | Long posed drives with per-frame dynamic boxes | Suburban; few dynamic agents | CC BY-NC-SA 3.0 · V |
| **Argoverse 1** | Ring cameras 30 Hz, stereo 5 Hz, LiDAR 10 Hz · V | Tracking 113 logs; forecasting ~325k × 5 s · R | Cuboid tracks per sweep (V); forecasting centroids at 10 Hz (R) | HD maps + 10 Hz tracks | Superseded by Argoverse 2 | CC BY-NC-SA 4.0 · V |
| **A2D2** (Audi) | 6 cameras, 5 LiDARs (R); CAN bus: speed, accelerations, yaw rate, steering, GPS (V) | 12,497 frames with 3D boxes; 392,556 unlabelled frames · V | 3D boxes, 14 classes, front-camera view only (R); no track IDs or velocity found | Commercial use allowed; rich ego dynamics | No tracks; "no derivatives" forbids publishing modified data | CC BY-ND 4.0 · V |
| **ApolloScape** (tracking + trajectory) | LiDAR + cameras at 10 fps · V | 53 + 50 min; 80k scans; 1,000 km of trajectories · V | Boxes + IDs and manual trajectories at 2 fps; many pedestrians · V | Dense mixed traffic | 2 Hz labels; short | Academic use only · R |
| **IDD-3D** (India) | Multi-camera + LiDAR · R | 12k annotated LiDAR frames, ~5 h · R | 3D boxes + instance IDs for tracking, 17 classes · R | Unstructured Indian traffic | Toolkit "coming soon" (V); access unclear | Not found |
| **DAIR-V2X / V2X-Seq** | Vehicle + roadside cameras and LiDARs (V); 10 Hz (R) | Perception: >15k frames / 95 scenes; forecasting: 672 h at 28 intersections · V | 3D boxes + persistent track IDs; trajectories + traffic lights at 10 Hz · R | Roadside view reveals occluded agents | China only | Licence not found |

Cityscapes is not a motion dataset: semantics on one frame per snippet, no tracks (V).

### 1c. Velocity sensing: radar and FMCW LiDAR
These measure velocity directly in one frame. **None is a NAVSIM input:** OpenScene lists 8 cameras and 5 LiDARs, no radar (V). Use them for pretraining, supervision or deployment studies.

| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **nuScenes radar** | 5 radars (V), ~13 Hz (R) | ≤125 returns per sweep · R | Per-return vx, vy and ego-compensated vx_comp, vy_comp; dynamic state (moving, oncoming, crossing, stopped…) · V | Direct 2D velocity per return, in the most-used dataset | Sparse returns | nuScenes terms (§1b) |
| **View-of-Delft (VoD)** | 3+1D radar ~13 Hz with radial velocity (raw + ego-compensated), HDL-64 ~10 Hz, stereo camera ~30 Hz · V | 8,600 frames, 123k+ boxes · V | 3D boxes, 13 classes, moving / stopped / parked; track IDs as a separate download · V | Radar velocity + tracks; urban, many cyclists | Small; front view ±32°, labels ≤ 50 m | Non-commercial; academic staff and students only · V |
| **MAN TruckScenes** | 4 cameras, 6 LiDARs, 6 4D radars (per-point vrel x/y/z), IMUs, GNSS · V | >740 scenes × 20 s · V | 3D boxes, 27 classes, tracked within scenes, 2 Hz keyframes; velocity by finite difference · V | 360° radar velocity; labels beyond 230 m | Truck-mounted; terminal scenes; 2 Hz labels | CC BY-NC-SA 4.0 · V |
| **aiMotive Multimodal** | LiDAR, 4 cameras (2 fisheye), front + back radar with per-point speed · V | 176 scenes, ~85 GB · V | 3D boxes with persistent IDs and per-object relative velocity XYZ · V | Direct relative-velocity labels; highway, night, rain | Small; ego motion not documented | Non-commercial research · V |
| **K-Radar** | 4D radar tensor with a Doppler axis, 2 LiDARs, 4 stereo cameras, RTK-GPS/IMU (V); ~10 Hz (D) | 58 sequences, 35k frames, ~15 TB · V | Hand-made 3D boxes; track IDs in the v2.0 labels · V | Full Doppler; fog, rain, snow | Huge raw data | CC BY-NC-ND · V: no derivatives |
| **RadarScenes** | 4 × 77 GHz 2D radars: range, azimuth, RCS, radial velocity (raw + compensated); odometry · V | 158 sequences, >4 h, ~119M detections · R | Class + track ID per detection, manual (R); no boxes · V | Doppler + instance tracks of moving road users | Radar only; no boxes, no elevation | Non-commercial · R |
| **Dual-Radar** | Two 4D radars at 20 Hz, 80-line LiDAR 10 Hz, camera 10 Hz · V | 151 sequences, 10,007 frames · V | 3D boxes + track IDs · V | Two radar types on the same scenes | Front only; small; no poses | Signed agreement · V |
| **AevaScenes** (FMCW LiDAR) | 6 FMCW LiDARs with per-point radial velocity, 6 × 4K cameras, all 10 Hz · V | 100 × 10 s = 10k frames · V | 3D boxes (25 classes), track IDs, per-object velocity vectors, per-point semantics · V | Measured velocity per point and per object, out to ~400 m | Tiny; one area, one vendor | Non-commercial licence · V |
| **HeLiPR / HeLiMOS** (FMCW) | FMCW + 3 other LiDARs · R | ~1 month of drives · R | HeLiMOS: auto-labelled moving / static per point · R | Real FMCW Doppler for motion segmentation | No boxes or tracks | Not found |

### 1d. Object-level motion (no raw sensors)
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **Waymo Open Motion (WOMD)** | Object tracks, roadgraph, signal states; LiDAR + camera embeddings for the first 1 s, no raw images (V) | 103,354 scenes (V); 10 Hz; windows of 1.1 s history + 8 s future (R) | Auto-labelled tracks (R); text via WOMD-Reasoning (§1g) | Largest diverse motion corpus; Waymax closed loop | Short history; no raw images | Waymo non-commercial · V |
| **Argoverse 2 Motion Forecasting** | Object tracks + HD map | 250k × 11 s, 6 cities (V); 10 Hz, 5 s history + 6 s future (R) | Tracks with inferred types; method not found · V | Longest standard history | No sensors | CC BY-NC-SA 4.0 · V |
| **Lyft / Woven L5 Prediction** | Agent tracks, traffic lights, semantic + aerial maps | 1,000 h (V); 170k × 25 s, 10 Hz (R) | Onboard-perception tracks · V | Volume | One fixed route (R); online labels; devkit archived Oct 2025 (V) | CC BY-NC-SA 4.0 · R |
ApolloScape and V2X-Seq trajectories are listed in §1b.

### 1e. Occupancy and flow labels
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **OpenScene occupancy (+ flow)** | Occupancy on nuPlan frames, 2 Hz · V | v1.0 labels: mini 6.7 GB, trainval 95.4 GB; none for test · V | 0.5 m voxels within ±50 m, built from 20 s of accumulated LiDAR (V). Per-voxel flow is claimed and a flow path is in the metadata (V), but the flow files are not verified (D) | On NAVSIM's own logs. Trainval covers navtrain, so no navtest is needed (V) | v1.0 only; some v1.1 frames unlabelled; class list marked "TODO" · V | CC BY-NC-SA 4.0 + nuPlan terms · V |
| **Occ3D** (nuScenes, Waymo) | Voxel labels from accumulated LiDAR + human labels · V | nuScenes 1,000 scenes; Waymo 1,150 · V | Semantics + visibility masks; no flow · V | Standard occupancy benchmark | No motion; nuScenes at 2 Hz | Source terms; generated data MIT or Apache (the README conflicts) · V |
| **OpenOccupancy** | Dense nuScenes occupancy · V | 28,130 + 6,019 frames · V | 0.2 m semantic voxels; no flow · V | Densest nuScenes occupancy | No motion | Code Apache-2.0 (V); label licence not found |
| **Waymo Occupancy Flow** | BEV grids rendered from WOMD tracks · V | All WOMD v1.1 scenarios · D | Future occupancy of observed and occluded vehicles + flow (dx, dy), 8 waypoints · V | Dense future-motion targets, including occluded agents | Vehicles only in the metric; not the nuPlan domain | Waymo non-commercial · V |
| **Argoverse 2 scene flow** | LiDAR sweep pairs 0.1 s apart · V | Argoverse 2 Sensor train/val · V | Per-point flow + dynamic flag; pseudo-labels from tracked cuboids · V | 10 Hz per-point motion; public challenges (R) | Rigid-box assumption; LiDAR only | CC BY-NC-SA 4.0 · V |

### 1f. Large camera + ego-trajectory sets (few or no object labels)
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **NVIDIA PhysicalAI-AV** | 7 cameras ~30 fps, LiDAR, radar on some clips, egomotion ~100 Hz · V | 1,700+ h, 306k clips, 25 countries · V | Machine labels "coming soon"; chain-of-causation reasoning labels being added · V | Scale, diversity, high-rate video, radar | Labels not shipped; LiDAR up to ~60 TB (D); restrictive licence | NVIDIA AV licence, gated (V); AV development "using NVIDIA technology" only (R) |
| **WOD-E2E** | 8 cameras (R); ego past 4 s / future 5 s at 4 Hz (V) | 4,021 runs (V), ≈12 h (R) | No boxes; up to 3 rater-scored trajectories · V | Rare long-tail events; preference labels; an external evaluation | No object labels; 4 Hz; no map | Waymo non-commercial · V |
| **L2D** (Yaak × Hugging Face) | 6 cameras, GPS, IMU, CAN at 10 Hz; OSM route waypoints · V | 100k episodes, 500+ h (release R3) · V | No 3D objects (D); natural-language instructions; expert vs learner tags · V | Permissive; route-conditioned; CAN signals | No object labels; includes learner drivers | Apache-2.0 · R |
| **CoVLA** | Front camera, CAN, GNSS, IMU · R | 10k clips, 80+ h, Tokyo · R | Automatic captions + trajectories · R | Same input shape as our pilot | Left-hand traffic; derivatives restricted | Custom non-commercial · R |
| **OpenDV-YouTube** | Video only | 1,700+ h · V | None | Video pretraining at scale | No ego state, no labels | Self-download from YouTube · V |
| **BDD100K** (video, 2D tracking) | 100k dashcam videos, 40 s at 30 Hz; GPS/IMU · V | >1,000 h, 1.8 TB (V); tracking subset 2,000 videos (R) | 2D boxes + IDs at 5 Hz in the tracking subset · V | Scale and diversity; 2D tracks | 2D only; monocular; no metric velocity | Research and non-profit use; commercial for members · V |

### 1g. Reasoning and language
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **ReCogDrive NAVSIM QA** | QA on navtrain's front camera | 752k QA · V | Scene description, influential objects, behaviour narration, decision explanation · V | On our exact training samples | Written by Qwen2.5-VL-72B; hindsight explanations (decision taken from the true future); QA form · V | Academic use · V. navtrain only (V) |
| **nuReasoning** | See §1b | 17k training clips · V | Spatial (~1 Hz), decision + counterfactual (every 5 s) · V; Gemini + human check (R) | Richest decision reasoning; human-checked | Mostly multiple-choice answers | See §1b |
| **NuPlanQA-1M** | QA on nuPlan cameras | 977,395 train QA · V | Perception, spatial and ego-centric QA, by GPT-4o · R | nuPlan domain; large | Perception-style, not reasoning | No licence file · V. Verify logs |
| **Impromptu-VLA** | QA + trajectories over 8 datasets, NAVSIM included | 80k clips from 2M+ · R | Scene description, lights, decision explanation, meta-actions · V | Unstructured corner cases | Mixed sources and terms | CC BY-SA 4.0 + source terms (V). Drop its navtest set (V) |
| **AutoDrive-P³** | Structured perception → prediction → planning on NAVSIM | Not found | Boxes, trajectories, structured answers; free text not distributed · V | NAVSIM domain | No free-text reasoning | Not stated · V. Its test split is navtest (V) |
| **Nu-X** | Narration + reasoning on nuScenes keyframes | 28,130 + 6,019 · V | Human, ~10 words per line · V | Human-written; MVLAD-AD uses it | Short; nuScenes only | No licence file (V); nuScenes terms (R) |
| **BDD-X** | Description + explanation on dashcam video | 6,970 videos, 77 h, 26k actions · V | Human · V | Human "what / why" style reference | Dashcam; no 3D | BDD100K terms · R |
| **DriveLM** (nuScenes; CARLA) | Graph QA: perception → prediction → planning | nuScenes: 30k frames / 360k QA (V); CARLA: ~71k keyframes (R) | Human-chosen keyframes, generated questions; CARLA rule-based · V | Structured reasoning chain | Templated | CC BY-NC-SA 4.0 · V |
| **LingoQA** (Wayve) | Video QA, front camera | 152.5k + 267.8k QA; eval 1,000 · V | Driver labels expanded by GPT-3.5 · R | Human-grounded "why" questions; an eval set | London only | Wayve non-commercial · V |
| **WOMD-Reasoning** | QA on Waymo motion data, no images | ~3M QA · V | Interaction and intention reasoning, by ChatGPT · V | Reasons about other agents' motion, as motion tokens would | No images | Waymo terms · V |
| **OmniDrive** | QA on nuScenes | Not found | Description, traffic rules, 3D grounding, counterfactuals, planning · V | Counterfactual reasoning | GPT-4-written · R | NVIDIA non-commercial · V |
| **NuInstruct** | Video QA on nuScenes | 91k QA, 17 subtasks · V | SQL-generated · V | Broad task coverage | Templated | Not found |
| **Smaller or partly released** | nuScenes-QA (460k template QA, R); Reason2Drive (600k pairs, only mini released, V); DriveLMM-o1 (18k / 4k, R); DRAMA (17,785 risk scenarios, R); Rank2Tell (R); Talk2Car (11,959 commands, R); DriveAction (16,185 QA, R); NVIDIA chain-of-causation labels (1,450 training clips, R) | | | | | Mostly non-commercial (R) |

**Labellers (code, not data):** AutoVLA's 4-step chain-of-thought prompt (V); NVIDIA's chain-of-causation auto-labeller (Apache-2.0, built on trajdata, which reads nuPlan; V). **Not released:** AutoVLA's nuPlan chain of thought, AdaThinkDrive, the AutoDrive-P³ / AUGUstus free-text traces (V).

### 1h. Simulation and synthetic training data
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **Bench2Drive** (CARLA) | 6 cameras, LiDAR, radar; privileged boxes with IDs and speed · V | 1k clips (400 GB) to 13,638 clips (4 TB); logged at 10 Hz · V | Exact boxes + speed for every actor · V | Closed-loop evaluation; latency injection (B2D-Robust); VLM bridge (B2D-VL) | CARLA visuals (sim-to-real gap, D); short routes near ceiling | CC BY-NC-ND · V: no derivatives |
| **PDM-Lite** (CARLA) | CARLA sensors + expert labels | ~364 GB (V); 581,662 frames at 4 Hz (R) | Privileged · R | Permissive; strong models trained on it | CARLA | CC BY 4.0 · V |
| **LEAD** (CARLA) | RGB 4 Hz; LiDAR, radar and boxes 20 Hz · V | 8,930 routes, 43 scenario types, ~1 TB · V | Boxes at 20 Hz · V | High-rate privileged tracks | CARLA; data licence not found | Code MIT · V |
| **SimLingo / LMDrive / DriveCoT** | CARLA + language | LMDrive ~64k clips (V); DriveCoT 36k samples (R) | Commentary, instructions, chain of thought · R | Language in the closed loop | CARLA | LMDrive CC BY-NC 4.0 (V); others not found |
| **SimScale** | Synthetic NAVSIM-style scenes, pseudo-expert labels | 384K samples, 2 Hz, 6 s clips (V); ~3.3 TB (D) | Pseudo-expert trajectories · V | Authors report +5.1 to +8.6 navhard EPDMS · R | Source split not stated | CC BY-NC-SA 4.0 · V. Check source logs |
| **Cosmos-Drive-Dreams** | Synthetic video | 81,802 videos from 5,843 labelled clips · V | Not checked | Visual augmentation; permissive | Open loop only | CC BY 4.0 · V |

### 1i. Long-tail and accident video
| Dataset | Data type | Scale · rate | Labels: motion · text | Strengths | Weaknesses | Licence · leakage |
|---|---|---|---|---|---|---|
| **DoTA** | YouTube dashcam video, 10 fps · V | 4,677 videos (R), 55 GB (V) | Anomaly start / end + class (V); tracklets of anomalous objects (R) | Long-tail "when / where / what" | Monocular web video; no metric motion | Not found |
| **DADA-2000** | Dashcam accident clips · R | 2,000 clips, 54 accident types · R | Driver eye fixations, accident window, crash-object location · R | Attention + accident timing | No 3D, no velocity | Not found |
| **CCD** (Car Crash Dataset) | YouTube crash + BDD100K normal clips, 10 fps · V | 1,500 crash + 3,000 normal clips · V | Per-frame accident flag, ego involvement, weather; detector boxes without IDs · V | Ego-involvement and onset labels | 5 s clips; 2D automatic boxes | Not found |
| **Nexar Collision Prediction** | Dashcam video, ~40 s clips · R | 1,500+ clips · V | Collision / near-collision / normal, with event and alert times · R | Real ego near-misses with alert timing | Video-level labels only | Open, with ethical-use limits · R |

## 2. Closed-loop simulators (evaluation tools, not datasets)
| Simulator | Agent sees | Rate | Strengths | Weaknesses | Terms |
|---|---|---|---|---|---|
| **nuPlan closed loop** | Tracked objects + 2 s buffer, ego, map, lights (V); cameras log-replayed only (D) | DB 20 Hz, sim 10 Hz, 15 s scenarios · V | NAVSIM's domain; native object input; IDM-reactive vehicles | Pedestrians open-loop; needs the full DBs; no release since 2023 · V | Devkit Apache-2.0; data non-commercial · V |
| **CARLA + Bench2Drive** | 6 cameras, LiDAR, radar; privileged boxes · V | 20 Hz · V | Latency injection of 100–500 ms (B2D-Robust) · V | Short routes near ceiling (LEAD 95.6 DS, R); also report the long Town13 routes (D) | See §1h · V |
| **AlpaSim** (NVIDIA) | Rendered cameras, ego, route; no agent tracks · V | 10 Hz decisions, asynchronous clocks · V | Camera closed loop on NAVSIM scenes | Its nuPlan track evaluates on **navtest** scenes (V): evaluation only; traffic off by default | Code Apache-2.0 · V |
| **NavSafe-∞** | Rendered cameras · V | Sim 10 Hz; policy at 2 Hz by default · V | 280 safety scenarios · V | Built from nuPlan test logs: evaluation only · V | Code Apache-2.0 · V |
| **Waymax / GPUDrive** | Objects only · V | 10 Hz · V | Fast object-level RL | No sensors | Waymax non-commercial; GPUDrive MIT · V |
| **NAVSIM v2 pseudo-sim** | Real + synthetic frames · V | 2 Hz; one 4 s plan per stage · V | The official metric | Cannot test re-planning at 10–20 Hz | Apache-2.0 · V |

**Poor fit, skipped:** HUGSIM (4 Hz, V); ScenarioNet / MetaDrive (object-level, V).

## 3. Licences and NAVSIM leakage
- **Non-commercial is the norm** (D: fine for papers; a product would need separate licences):
  - CC BY-NC-SA 4.0: nuPlan / OpenScene (V), Argoverse 2 (V), ONCE (V), nuScenes (R).
  - Waymo (V), nuReasoning's free tier (V), Bench2Drive (NC-ND, V), K-Radar (NC-ND, V), View-of-Delft (academic staff and students only, V).
- **ShareAlike** probably extends to derived data we publish, such as token sets and labels (D).
- **Permissive:** ZOD (CC BY-SA 4.0, V), PDM-Lite (CC BY 4.0, V), L2D (Apache-2.0, R), Cosmos-Drive-Dreams (V), A2D2 (CC BY-ND 4.0, V; no modified releases).
- **Restrictive:** PhysicalAI-AV (R). Read the full licence before relying on it.
- **Teacher outputs carry their own terms.** Qwen's licence asks for "Built with Qwen" on distributed models trained on its outputs (R). Gemini and GPT terms not checked.
- **NAVSIM rule:** never train on navtest / navhard / warmup / private splits (V). Concretely:
  - exclude OpenScene `test` logs when pulling raw nuPlan;
  - drop Impromptu-VLA's navtest set;
  - AlpaSim's nuPlan track and NavSafe-∞ are built on test logs (V): evaluation only;
  - check the source logs of SimScale, nuReasoning and NuPlanQA by log name.

## 4. Implications [proposed]
1. **D6:** navtrain stays the core of v1.
2. **D6, next source: the raw nuPlan DBs for the same logs.**
   - Same scenes, but at 20 Hz, with whole-log history, route and traffic lights. Same licence; no domain shift (D).
   - Serves C1 history length/rate studies, the motion encoder (e), the refresh loop and, later, the nuPlan closed loop.
   - Joining on log name + timestamp is to be verified in P2. The DBs are 1.22 TB for all splits (V).
3. **D9:** add ReCogDrive's navtrain QA as a comparison arm in the P5 teacher pilot. It is free, but model-written and hindsight. Borrow structure from the chain-of-causation and AutoVLA prompts.
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
  - `github.com/haha-yuki-haha/AutoDrive-P3`.
  - `github.com/Robot-K/Hint-AD`: release v1.0.
  - `github.com/JinkyuKimUCB/BDD-X-dataset`.
  - `github.com/ucla-mobility/AutoVLA`: `dataset_utils/preprocessing/cot_prompts.py`.
  - `github.com/NVlabs/alpamayo-coc-autolabeler`.
  - `github.com/OpenDriveLab/DriveLM`, `github.com/xmed-lab/NuInstruct`, `github.com/NVlabs/OmniDrive`, `github.com/wayveai/LingoQA`, `github.com/yhli123/WOMD-Reasoning`, `github.com/fudan-zvg/reason2drive`.
- **Sensor and motion data:**
  - `github.com/nutonomy/nuscenes-devkit`.
  - `github.com/argoverse/user-guide` (datasets, terms).
  - `github.com/waymo-research/waymo-open-dataset` (LOG.md, protos).
  - `github.com/woven-planet/l5kit`.
  - `github.com/zenseact/zod`.
  - `github.com/scaleapi/pandaset-devkit`.
  - ONCE: `github.com/once-for-auto-driving/once-for-auto-driving.github.io`.
  - `github.com/NVlabs/physical_ai_av` (wiki).
  - `github.com/huggingface/blog` (`lerobot-goes-to-driving-school.md`).
  - `github.com/stepankonev/StandardE2E` (KITScenes).
  - `github.com/OpenDriveLab/DriveAGI` (OpenDV).
- **Simulation:**
  - `github.com/Thinklab-SJTU/Bench2Drive` and `Bench2Drive-Robust`.
  - `github.com/autonomousvision/carla_garage` (leaderboard_2).
  - `github.com/kesai-labs/lead`, `github.com/RenzKa/simlingo`, `github.com/opendilab/LMDrive`.
  - `github.com/NVlabs/alpasim` (`e2e_challenge/README.md`).
  - `github.com/VAIL-UCLA/NavSafe-infty`.
  - `github.com/waymo-research/waymax`, `github.com/Emerge-Lab/gpudrive`.
  - `github.com/OpenDriveLab/SimScale`.
  - `github.com/nv-tlabs/Cosmos-Drive-Dreams`.
- **Radar and FMCW:**
  - `github.com/tudelft-iv/view-of-delft-dataset` (docs).
  - `github.com/TUMFTM/truckscenes-devkit` (schema, data classes).
  - `github.com/aimotive/aimotive_dataset` and `aimotive-dataset-loader`.
  - `github.com/kaist-avelab/K-Radar`.
  - `github.com/oleschum/radar_scenes`.
  - `github.com/adept-thu/Dual-Radar`.
  - `github.com/aevainc/aevascenes`.
  - nuScenes radar fields: `nuscenes-devkit/python-sdk/nuscenes/utils/data_classes.py`.
- **Occupancy and flow:**
  - `github.com/Tsinghua-MARS-Lab/Occ3D`.
  - `github.com/JeffWang987/OpenOccupancy`.
  - OpenScene `docs/getting_started.md` and `docs/dataset_stats.md`.
  - Waymo `occupancy_flow_metrics.proto` and the occupancy-flow tutorial.
  - `github.com/argoverse/av2-api` (`evaluation/scene_flow`).
- **Classic driving:**
  - `github.com/autonomousvision/kitti360Scripts`.
  - `github.com/bdd100k/bdd100k` (docs).
  - `github.com/argoverse/argoverse-api`.
  - A2D2 bucket README and LICENSE (`aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com`).
  - `github.com/ApolloScapeAuto/dataset-api`.
  - `github.com/shubham1810/idd3d_kit`.
  - `github.com/mcordts/cityscapesScripts`.
  - `github.com/AIR-THU/DAIR-V2X` and `DAIR-V2X-Seq`.
  - AWS open-data registry entries (KITTI, Argoverse, A2D2, TruckScenes).
- **Accident video:**
  - `github.com/MoonBlvd/Detection-of-Traffic-Anomaly`.
  - `github.com/JWFangit/LOTVS-DADA`.
  - `github.com/Cogito2012/CarCrashDataset`.
  - `github.com/getnexar/BADAS-Open`.
