# Getting the motion of surrounding objects (2026-10-20)

**Status: research input [proposed].** This answers the researcher's question of 2026-10-20: how can the pilot know the relative speed and motion of the objects around it, and is "open up NAVSIM and track it ourselves" viable? Nothing here is adopted.

**Related docs:**
- Datasets: `docs/training_data/DATASETS.md`.
- Claims (C1): `docs/RESEARCH_DIRECTION.md`.

**Evidence:** V / R / D as in `docs/RESEARCH_DIRECTION.md`.
- Latencies measured by NVIDIA or Ultralytics in their own repos count as vendor benchmark files: V.
- Accuracy numbers and scores are self-reported: R.
- arXiv was blocked, and paper text was read only through third-party copies, so paper-text facts are R.

## 1. Short answer
- **Yes: track it ourselves on NAVSIM.** NAVSIM already has the labels to train a 2 Hz detector and tracker. Every frame of a navtrain scene, history included, carries boxes, class, velocity and persistent track IDs (V).
- **The hard limit is NAVSIM's test input, not training data.**
  - A scored tracker sees only 4 frames at 2 Hz, a 1.5 s span (V).
  - For the 2025 challenge on navhard two-stage, the maintainers gave **no LiDAR for any test frame**. They told participants to "focus on camera-only methods" (V, NAVSIM issues #63 and #84).
  - So plan a **camera-only** tracker for v2 scoring (D).
- **2 Hz is workable.** It is the nuScenes keyframe rate.
  - Camera streaming detectors (StreamPETR, Sparse4D v3) fuse 2 Hz keyframes natively (V, code). They report velocity errors of 0.21–0.24 m/s (R).
  - In SimpleTrack, tracking on keyframes only matched 10 Hz tracking: AMOTA (the nuScenes tracking score) 0.687 vs 0.687 (R).
  - Caveat: nuScenes trackers keep memory across 20 s scenes, while NAVSIM gives 1.5 s (D).
- **High-rate vision data already exists for the same logs.**
  - nuPlan's raw sensors (cameras 10 Hz, LiDAR 20 Hz) cover all 1,192 navtrain logs: 19.06 TB in all, and readable one file at a time (V).
  - It is needed only for a 10–20 Hz deployment tracker.
- **Motion tokens are cheap at inference.** On Thor, 20–60 motion tokens cost about 0.4–4 ms of prefill. One extra image frame costs about 17–27 ms (D, from the TensorRT Edge-LLM numbers in `RESEARCH_DIRECTION.md` §2.3).
- **But on NAVSIM v1, leading methods do without explicit motion.**
  - None feeds in other agents' motion at test time. They learn its consequences through PDM-score supervision or RL (§4).
  - History adds only +0.1 to +2.7 PDMS (R).
  - No evidence either way on navhard.

## 2. Facts that constrain the design
| Fact | Mark |
|---|---|
| **Test input:** 4 frames at 2 Hz, each with 8 cameras, merged LiDAR (v1) and ego status (pose, velocity, acceleration, command). No annotations | V |
| **No LiDAR on navhard / v2 test frames:** "For this CVPR challenge, we will not be providing LiDAR point clouds for any test frames (real or synthetic)" (13 Mar 2025). Also: "The current metric does not support LiDAR-based methods such as Transfuser" (4 Apr 2025). Not checked: whether the real first-stage frames in local navhard evaluation include LiDAR | V |
| **LiDAR per frame:** one merged sweep of the 5 LiDARs, ~93k points, no per-point time offsets. So there is no 0.5 s multi-sweep stack of the kind nuScenes detectors use to read velocity (D) | V |
| **Training labels:** every navtrain frame, history included, has boxes, class, `velocity_3d` (a network estimate) and track tokens. Velocities are stored in each frame's own ego frame | V |
| **The only public NAVSIM-trained detector found** is the TransFuser / LTF agent head: vehicles only, current frame only, within ±32 m, up to 30 boxes `[x, y, heading, length, width]`. No velocity, height or ID. LTF is camera-only | V |
| **Occupancy + flow:** OpenScene v1.0, mini and trainval only (95.4 GB). 0.5 m voxels with a per-voxel velocity `(vx, vy)`. The label code is not released | V |
| **Radar:** none in nuPlan or OpenScene | V |
| **Raw full-rate sensors** (Motional's public bucket): train 5.39 TB cameras + 8.60 TB LiDAR; val 0.87 + 1.42; test 0.64 + 1.05; DBs 1.22 TB. Per-file range reads work. nuPlan's terms apply | V |
| **Coverage:** all 1,192 navtrain logs are in the train/val manifests. All 136 navtest and 76 navhard logs are in the test manifest, so never train a tracker on the test blobs | V |
| **Oracle evaluation:** local scoring passes the full `Scene` to agents with `requires_scene=True`, but submission creation refuses them. Oracle-motion runs are fine locally, never on a leaderboard | V |

## 3. Methods
| | Method | Gives | Scorable on v1 / v2 | Training data | Accuracy evidence | Edge evidence | Effort (D) |
|---|---|---|---|---|---|---|---|
| **A** | **Oracle motion from annotations** | Boxes, velocity, tracks over the 4 frames | Local only (V) | navtrain annotations | Upper bound. Optimistic: the scorer's world is built from the same auto-labels (D) | n/a | ~1 week |
| **B** | LiDAR detector (CenterPoint / PointPillars) on 2–4 sweeps + greedy 2 Hz tracker | Boxes, velocity, IDs | ✓ / ✗ | navtrain LiDAR with history + labels | CenterPoint, 10 sweeps, nuScenes: AMOTA 63.7, velocity error 0.288 m/s (R). With 2 sweeps 0.5 s apart: not found | DRIVE Orin: CenterPoint 35.7–40 ms, PointPillars 6.8 ms (V). Thor: PointPillars + tracker 21 ms (R) | 3–5 weeks |
| **C** | **Camera streaming detector + tracker** (Sparse4D v3 / StreamPETR) trained on navtrain at 2 Hz | Boxes, velocity, IDs | ✓ / ✓ | navtrain cameras with history (445 GB) + labels | nuScenes: velocity error 0.21–0.24 m/s, AMOTA 0.48–0.65 (R) | Orin: PETRv2 (2 frames) 55.8 ms, UniAD-tiny 39 ms (V). Thor: not found | 4–8 weeks |
| **D** | **NAVSIM's LTF head on each frame + greedy tracker**; later retrained with all classes + velocity | Vehicle boxes; velocity from position differences | ✓ / ✓ | Released checkpoint, then navtrain | No published accuracy | Not found | Days; then 2–3 weeks |
| **E** | 2D detector + ByteTrack on the front camera | 2D tracks; metric speed only with depth | ✓ / ✓ | COCO + projected navtrain boxes | Association is designed for 30 fps video and breaks at 0.5 s gaps (R) | YOLO26 on Thor: 1.4–2.6 ms (V) | 1–2 weeks; weak |
| **F** | More frames into the VLM (or cached features) | Implicit motion | ✓ / ✓ | Pilot data | DriveVLA-W0: +2.7 PDMS with a frame 1 s back (R) | +17–27 ms per frame on Thor (D) | 1–2 weeks. This is C1's comparison arm, not a motion source |
| **G** | Occupancy-flow network on OpenScene labels | Dense per-voxel velocity | ✓ / ✓ | OpenScene v1.0 labels | No baseline found | Not found | 6–10 weeks |
| **H** | Scene flow / optical flow across 0.5 s | Per-point or per-pixel flow | ✓ / LiDAR ✗ | Self-supervised | Learned scene flow fails beyond 0.3–0.4 s gaps (R) | Not found | 3–6 weeks; weak |
| **I** | **Privileged teacher → sensor student** | Motion-aware behaviour, no motion input at test time | ✓ / ✓ | Teacher trained on A | DriveZero's camera student: 95.3 PDMS, 57.1 navhard EPDMS (R; another source gives 94.8) | Same as the pilot | 3–6 weeks after A |
| **J** | Radar Doppler / FMCW LiDAR | Measured radial velocity per point | ✗ / ✗ | nuScenes, View-of-Delft, TruckScenes, aiMotive, PhysicalAI-AV, AevaScenes (`DATASETS.md` §1c) | CenterFusion (R) | n/a | Deployment only |
| **K** | Velocity regressed from a single frame | Velocity priors only | ✓ / ✓ | navtrain | Single-frame camera models: ~0.9 m/s velocity error (R) | n/a | Weak: ill-posed |

## 4. What published methods feed in, and what they train on
"t only" means the planner uses only the current frame of NAVSIM's 4. PDMS is navtest PDMS (v1). Every score is R.

| Method | PDMS (R) | Test-time input | How other agents enter | Training data |
|---|---|---|---|---|
| Ego-status MLP | 65.6 | Ego velocity, acceleration, command at t; no sensors · V | Not at all · V | navtrain · V |
| TransFuser / LTF (NAVSIM baselines) | 84.0 / 83.8 | 3 front cameras stitched + LiDAR BEV (LTF: no LiDAR), t only · V | Auxiliary loss on vehicle boxes (no velocity) and a BEV map · V | navtrain · V |
| PDM-Closed (privileged rule planner) | 89.1 | Ground-truth boxes + map · V | Ground-truth agents rolled forward at constant velocity · V | None · V |
| UniAD / PARA-Drive (NAVSIM ports) | 83.4 / 84.0 | Cameras; history not found · R | Tracking, motion and occupancy heads · R | navtrain · R |
| Hydra-MDP++ | 86.6–91.0 | 3 cameras at t and t−0.5 s · R | Temporal fusion; predicts PDM sub-scores · R | navtrain · R |
| DiffusionDrive | 88.1 | TransFuser input, t only · V | Detection + BEV aux losses; decoder attends to agent queries · V | navtrain · V |
| WoTE | 88.3 | 3 cameras + LiDAR, t only · V | World model predicts future agent boxes (+4 s) and BEV maps · V | navtrain · V |
| GTRS-Dense | navhard 41.7 | 3 cameras, t only · V | Only PDM sub-score heads; no boxes · V | navtrain + PDM scores for a 16,384-trajectory vocabulary · V |
| iPad | 91.1–91.7 | 4 cameras at t + ego history of all 4 frames · V | Map + prediction aux (R); agent-box targets commented out (V) | navtrain · V |
| DrivoR | 93.1–94.6 | 4 cameras at t, no LiDAR · V | None explicit; a scorer predicts PDM sub-scores · V | navtrain (+navval) + SimScale synthetic data · V |
| ReCogDrive (VLA) | 89.6–90.8 | Front camera at t + 4 ego poses as text · V | None at test time. QA pretraining built from boxes and velocities; RL on PDMS · R | 3.1M QA from 12 public sets + navtrain · R |
| AutoVLA (VLA) | 89.1 | 3 front cameras × 4 frames · V | Chain of thought naming crucial objects; RL on PDMS · R | nuPlan 166k + nuScenes 19k + DriveLM; RL on navtrain · R |
| DriveVLA-W0 (VLA) | 90.2 | Front camera at t and t−1 s + past action · R | Future-image world model as an aux loss · R | nuPlan pretraining + navtrain · R |
| WAM-Diff / WAM-Flow (VLA) | 91.0 / 90.3 | One front image + ego state + command · R | VQA co-training; RL on PDMS · R | 668k nuPlan trajectories + 0.8–6.5M VQA + navtrain · R |
| DriveFine (VLA, LLaDA-8B) | 90.7 | One front image · R | ReCogDrive QA; RL · R | ReCogDrive QA + NAVSIM RL · R |
| AUGUstus (VLA, Qwen3.5-2B) | 94.0–94.6 | 3 cameras stitched, 4 frames at 2 Hz as a video clip · V | Perception → prediction → planning chain of thought (2D boxes); RL · V | SFT + RL on navtrain. The README also claims 318k h of video pretraining · R |
| DriveZero | 95.3 | Camera-only; history not found · R | Student imitates a privileged RL teacher's rollouts · R | navtrain · R |
| Driving on Memory | 91.1 | **No live camera:** memory tokens retrieved by location, plus ego state · R | None · R | navtrain · R |
| ReflectDrive | 91.1 (94.7 with ground-truth agents) | 3 cameras + ego text · R | Safety check against constant-velocity obstacles · R | Not found |
| MVLAD-AD (nuScenes) | n/a | Front camera at t + 5 s of ego history as text · V | None · V | nuScenes + Nu-X · V |
| Alpamayo-R1 (NVIDIA) | n/a | 4 cameras × 4 frames at 10 Hz + 1.6 s of egomotion · V | Chain-of-causation reasoning about critical agents · R | Internal data (up to 2M segments) + 700k reasoning traces · R |
| PlanT (CARLA) | n/a | Object tokens with **speed** + route · V | Explicit per-object speed · V | CARLA expert data · V |

**What this shows (D unless marked):**
1. **Single frame is the norm.** Most released NAVSIM planners plan from the current frame only (V, code).
2. **No leading NAVSIM method feeds in other agents' velocities at test time.** Other agents enter through the training signal:
   - NAVSIM scores collisions and time-to-collision against the logged future boxes (V). So predicting PDM sub-scores (Hydra, GTRS, DrivoR) or running RL on the PDM score teaches the consequences of other agents' motion without perceiving it.
   - World-model targets: WoTE's future agent boxes.
   - A privileged teacher: DriveZero.
3. **History helps only a little on NAVSIM v1** (R):
   - Hydra-MDP++: +0.1 PDMS.
   - ResWorld: +0.7.
   - DriveVLA-W0: +2.7, with two frames 1 s apart.
   - Driving on Memory scores 91.1 without seeing the current scene. So v1 mostly rewards static layout plus ego state.
4. **Explicit motion pays off when something uses it directly** (R):
   - ReflectDrive's safety check: 84.8 → 91.1 with constant-velocity obstacles, and 94.7 with ground-truth agents.
   - PDM-Closed reaches 89.1 with ground-truth agents and no learning.
5. **No history or motion ablation on navhard was found.** That reactive two-stage test is where motion should matter most: an open gap, and a place for C1.

## 5. Recommended path [proposed] (D)
1. **Oracle gate first.** About 1 week to build, plus training runs.
   - **Build:** motion tokens from navtrain annotations:
     - 4 frames, ego-motion compensated;
     - velocity from track positions, with the network velocities used only as a check.
   - **Train:** a cheap planner with and without the tokens. For example, an object-token planner in the PlanT style, or the LTF baseline plus tokens.
   - **Score:** locally, with a `requires_scene` agent, on navtest. Also on navhard, if its synthetic stage carries annotations (not checked).
   - **Go/no-go for C1:** if oracle motion does not improve navhard EPDMS, NC or TTC over image-only at a matched token budget, a real tracker will not either.
   - **Reuses:**
     - the teacher for I;
     - labels for B–D;
     - a robustness curve: add noise, dropouts, ID switches and delay to the oracle tokens, then place each real tracker on that curve;
     - token-format ablations without perception noise.
2. **Track NAVSIM ourselves, camera-only:**
   1. **Days, no training:** run the released LTF head on each of the 4 frames.
      - Move the boxes into the current ego frame using AgentInput's poses.
      - Associate across frames with CenterPoint's greedy rule: shift each box by v·Δt, then gate per class.
      - Take velocity as the least-squares slope.
   2. **2–3 weeks:** retrain the head on navtrain with pedestrians and cyclists, a velocity output and previous-frame features.
   3. **Main deployable result:** a Sparse4D-v3-class streaming tracker trained at 2 Hz.

   Check each step against the oracle on held-out navtrain logs: tracking score, and velocity error per speed bin.
3. **LiDAR tracker (B):** a secondary arm only, for v1 navtest and LiDAR-equipped deployment.
4. **Distillation (I):** the "motion learned, not input" arm, once A exists. It suits a single-frame pilot.
5. **Deployment at 10–20 Hz, later:**
   - Retrain the same tracker on full-rate nuPlan sensors. Range-reading only the navtrain windows means about 1.3–3.9 TB of LiDAR (D). Radar is the alternative.
   - Thor latency for camera 3D trackers is unknown: measure it on the device.

## 6. Single frame without motion context: what it costs and what it keeps (D)
- **The evidence supports a single-frame pilot on NAVSIM v1** (§4). Single front-camera VLAs score about 90–91 PDMS (R).
- **Impact on the claims:**
  - **C1 as stated** (motion instead of past images) falls away if the pilot gets no motion at all.
  - **C2 alone faces strong prior work:** WAM-Diff and WAM-Flow already use 20,001 action tokens (V).
  - **C3 gains:** a single frame is the cheapest input.
- **Middle path:** one image frame **plus** motion tokens from a tracker still matches the original claim (motion instead of past images).
  - The single-frame pilot is C1's control arm anyway, so building it first loses nothing.
  - Decide on C1 after the oracle gate (§5.1). It costs about a week before any tracker work.

## Sources (primary unless marked; read 2026-10-20)
- **NAVSIM:**
  - `github.com/autonomousvision/navsim`:
    - `navsim/common/dataclasses.py`, `docs/agents.md`, `docs/splits.md`;
    - `navsim/agents/` (TransFuser / LTF, ego MLP, constant velocity);
    - `navsim/planning/script/run_pdm_score.py`, `run_create_submission_pickle.py`;
    - `pdm_observation.py`.
  - Issues #63 (13 Mar, 15 Apr 2025) and #84 (4 Apr 2025).
- **nuPlan:**
  - `github.com/motional/nuplan-devkit`: `docs/nuplan_schema.md`, `docs/dataset_setup.md`, `nuplan/database/utils/pointclouds/lidar.py`.
  - Public bucket listing and sensor manifests: `motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/`.
- **OpenScene:** `github.com/OpenDriveLab/OpenScene` (README, `docs/getting_started.md`, `docs/dataset_stats.md`, occupancy viewer).
- **Tracking and detection:**
  - nuScenes tracking and detection eval READMEs (`nuscenes-devkit`).
  - `github.com/tianweiy/CenterPoint` (configs, `pub_tracker.py`).
  - `github.com/tusen-ai/SimpleTrack` (`docs/nuScenes.md`).
  - `github.com/exiawsh/StreamPETR`, `github.com/HorizonRobotics/Sparse4D`, `github.com/fundamentalvision/BEVFormer`.
- **Edge:**
  - `github.com/NVIDIA-AI-IOT/Lidar_AI_Solution`, `github.com/NVIDIA-AI-IOT/CUDA-PointPillars`.
  - `github.com/NVIDIA/DL4AGX` (AV-Solutions: uniad-trt, petr-trt, vad-trt, far3d-trt).
  - Ultralytics `docs/en/guides/nvidia-jetson.md`.
  - `github.com/NVIDIA/TensorRT-Edge-LLM`.
- **Methods (code and READMEs):**
  - `hustvl/DiffusionDrive`, `NVlabs/GTRS`, `liyingyanUCAS/WoTE`, `Kguo-cs/iPad`, `valeoai/DrivoR`, `YvanYin/GoalFlow`.
  - `xiaomi-research/recogdrive`, `ucla-mobility/AutoVLA`, `haha-yuki-haha/AUGUstus`.
  - `XiaomiAutoL3/DriveZero`, `boschresearch/MemoryDrivoR`, `pixeli99/ReflectDrive`.
  - `NVlabs/alpamayo`, `autonomousvision/plant`, `lan-qing/MVLAD-AD`.
- **Paper text (R; third-party copies, since arXiv was blocked):** Hydra-MDP++, DriveVLA-W0, WAM-Diff, WAM-Flow, DriveFine, ReflectDrive, AdaThinkDrive, SimScale, ResWorld, LAW, ICP-Flow.
