# P1 requirements: data for the two small models (draft, 2026-10-20)

**Status: draft for Phase 1 of `README.md` §4 [proposed].**
- **Scope [agreed 2026-10-20]:** the fast pilot and the reasoner. Multi-frame and object-motion data are deferred.
- **Exit criterion:** you agree with §1–§4, or change them.
- **No code and no field definitions yet.** This fixes *what information* flows and what it looks like, not how it is stored.

Evidence marks (V / R / D) as in `docs/RESEARCH_DIRECTION.md`. The examples are illustrative and invented. Sources are named as in NAVSIM (e.g. CAM_F0); other datasets map onto the same information (§3).

## 1. What each model sees and produces

### 1a. Fast pilot (Qwen3.5, ~2B)
| Role | Information | What it looks like (illustrative) | Source | Visible at test time? |
|---|---|---|---|---|
| Input | Current front camera image | One image, ~250–500 visual tokens | CAM_F0 at t (V) | Yes (V) |
| Input, view option | Wider view: L0 + F0 + R0 stitched | One panorama, same token budget | The 3 front cameras at t (V) | Yes (V) |
| Input | Ego state | "Speed 8.2 m/s, acceleration −0.6 m/s²" | Ego status at t (V) | Yes (V) |
| Input | Route command | "Command: left". It is a 4-way one-hot; the prototype reads it as left / straight / right / unknown, to verify in P2 | Driving command (V) | Yes (V) |
| Input, view option | Ego history, 1.5 s | "Past positions (m): (−12.1, 0.4) (−8.0, 0.2) (−4.0, 0.1) (0, 0)" | The 4 ego poses (V) | Yes (V) |
| Input, optional | The reasoner's latest decision, possibly stale or missing | "Advice (2 s old): decelerate, follow lane" | In training: the rule-based decision from an earlier frame, randomly aged or dropped (D) | Comes from the reasoner |
| Target | Future path: 8 poses over 4 s at 0.5 s | 8 action tokens; the tokenizer is the C2 choice | Recorded future ego poses (V) | No: label |
| Later, for RL | PDM score of sampled paths | Nothing new to store | NAVSIM's metric cache built for navtrain (D: to confirm in P2) | No |

### 1b. Reasoner (Gemma 4 E2B/E4B)
| Role | Information | What it looks like (illustrative) | Source | Visible at test time? |
|---|---|---|---|---|
| Input | Current front image. More cameras or a larger image are possible, since it runs less often | One image | CAM_F0 (+ L0 / R0) at t (V) | Yes (V) |
| Input | Ego state + route command | As for the pilot | Ego status (V) | Yes (V) |
| Target 1 | Decision from a closed set: longitudinal × lateral | "Decision: decelerate, follow lane" | Rules applied to the recorded future. Starting point: prototype v0.1, 6 longitudinal × 8 lateral classes, thresholds calibrated in P2 | No: label |
| Target 2 | Reasoning: 1–2 sentences on why this decision | "A van is stopped in our lane ahead and the left lane is occupied, so slow down behind it." | Teacher VLM, run offline (P5) | No: label |
| Target 3 | Narration: 1–2 sentences on what matters in the scene | "Two-lane urban street; stopped van with hazard lights ahead; cyclist on the right." | Teacher VLM, run offline (P5) | No: label |

**Output order: decision, then reasoning, then narration (D).** The pilot can use the decision as soon as it is generated, and narration can be cut short to save compute (C3).

## 2. Rules for the labels [proposed]
1. **Visible-evidence rule for text.**
   - The reasoner sees one frame. So narration and reasoning may only state what that frame and the ego state show: positions, brake lights, turn signals, road layout.
   - They must not state other agents' speeds or future moves. Otherwise the student learns to make things up.
   - The teacher may see privileged information (boxes, the recorded future) to get facts right, but must not write what the student cannot see (D).
2. **Decision consistency.** The decision label comes from deterministic, versioned rules over the recorded future. Teacher text must support that decision; samples where it does not are rejected or flagged (D).
3. **Hindsight vs causal prompting stays open (D9).** The P5 pilot compares two teachers: one told the true decision (AutoVLA style) and one that must predict it.
4. **Version everything:**
   - decisions: the rules version;
   - text: the teacher model and prompt;
   - every release is immutable (`README.md` §2).
5. **Short by default:** about 40 tokens or fewer each for reasoning and narration. Output tokens cost decode time on the device (C3) (D).

## 3. Sources, scale and splits
**Principle [agreed 2026-10-20]:** train on several datasets, not navtrain alone, for both quality and quantity. NAVSIM stays the target for fine-tuning and evaluation [proposed].

A single-frame sample needs only three things, and no object labels (D):
- a front image;
- the ego trajectory around it: 1.5 s back and 4 s ahead, at ≥2 Hz;
- speed and acceleration.

So many datasets qualify.

### 3a. Candidate sources [proposed]
"Frames at 2 Hz" is hours × 7,200 (D). Clips shorter than ~20 s lose another ~25–30% to the 5.5 s window (D).

| Source | Frames at 2 Hz | Pose quality | Command / route | Licence | Proposed role |
|---|---|---|---|---|---|
| **NAVSIM navtrain** | ~103k samples (V) | nuPlan localization (V) | Route-based command (V) | Non-commercial (V) | Target domain: fine-tuning, dev split |
| **Rest of OpenScene trainval:** the same 1,192 logs, minus the scenes NAVSIM keeps for navtrain as "non-trivial" (V) | ~0.8M (D, ~120 h). Sensors 2.1 TB vs 445 GB for navtrain (V) | Same | Same | Same | Same-domain bulk; down-weight easy cruising |
| **nuScenes** | ~34k keyframes (V) | Localization + CAN bus (V) | Derived from the future path (D) | Non-commercial (R) | Small, high quality; MVLAD-AD comparison |
| **Waymo Perception / WOD-E2E** | ≤80k / ~90k (D) | Pose per frame (V) / ego at 4 Hz (V) | WOD-E2E has an intent field (V) | Waymo non-commercial (V) | High quality; long tail (E2E); an external check |
| **nuReasoning** | ~0.6M in train (D, 17k × 20 s) | Not checked | Route commands (V) | Non-commercial + commercial terms (V) | NAVSIM-like at 10 Hz, plus reasoning. Check log overlap first |
| **NVIDIA PhysicalAI-AV** | ~12M (D) | Egomotion at ~100 Hz (V) | Not found | Restrictive (R) | Scale and diversity (25 countries), if the licence fits |
| **L2D** | ~3.6M (D) | GPS + IMU at 10 Hz (V) | OSM route waypoints (V) | Apache-2.0 (R) | Permissive bulk; drop learner-driver episodes (tagged, V) |
| **CoVLA** | ~0.6M (D) | CAN + GNSS (R) | Not found | Custom non-commercial (R) | Front camera + captions; left-hand traffic |
| **ZOD** | ~60k from sequences (D), plus long drives | GNSS/IMU at 100 Hz (R) | Not found | CC BY-SA 4.0 (V) | Europe; permissive |
| **ONCE** | ~1M (D) | Pose per frame (V) | Not found | Non-commercial + extra terms (V) | China; bulk pretraining |
| **BDD100K** | >7M (D) | Phone GPS/IMU, rate not found (V) | None | Research use (V) | Images only: GPS is likely too noisy for 0.5 s waypoints (D) |
| **SimScale** (synthetic) | 384K samples (V) | Pseudo-expert labels (V) | NAVSIM-style | CC BY-NC-SA 4.0 (V) | NAVSIM-style augmentation. Check its source logs first |

Facts and licences per dataset: `DATASETS.md`. What one moment looks like in each source, and how it maps to one canonical frame: `DATA_ATLAS.md`.

### 3b. Rules for mixing sources [proposed] (D unless marked)
1. **One canonical sample format for every source.** Each sample holds:
   - the front image and its intrinsics;
   - ego poses in the rear-axle frame (x forward, y left), with timestamps;
   - speed and acceleration;
   - the command, with its definition recorded;
   - provenance: source, licence, split.

   Each source's ego-frame origin is checked in P2.
2. **One command definition.**
   - Calibrate a rule on navtrain that reproduces NAVSIM's command from the path, and apply it to sources without routes.
   - Record whether each command is route-based or derived.
3. **Camera differences [agreed 2026-10-27]: keep native images.** The aim is a general-purpose model that learns every camera.
   - Record intrinsics, distortion and field of view.
   - Tag the camera in the prompt.
   - Rectify to one view only if results are poor.
4. **Quality filters:**
   - pose jumps and implausible jerk;
   - image–pose time sync;
   - **project the future path into the front image** and check that it follows the road. This is a cheap visual check that works on every source;
   - drop learner drivers;
   - drop near-duplicate frames: sample at ≤2 Hz and skip long standstills.
5. **Balance** by source, manoeuvre (the rule-based decision classes), speed and city. Down-weight straight cruising and standstill.
6. **Training stages:**
   1. pretrain on the mixture;
   2. fine-tune on navtrain;
   3. RL on navtrain with the PDM score.

   This mirrors published VLAs, which pretrain on more than navtrain (R):
   - WAM-Diff / WAM-Flow: 668k nuPlan trajectories + VQA;
   - AutoVLA: nuPlan + nuScenes, then RL on navtrain.
7. **Measure each source's value before scaling downloads.** Run add-one-source ablations, scored on the navtrain dev split.
8. **Splits:**
   - per-source train / dev by log;
   - each dataset's official test split stays unused;
   - NAVSIM's navtest / navhard / warmup splits never;
   - exclude nuPlan test logs from any raw-nuPlan pull (V);
   - check nuReasoning and SimScale for overlap.
9. **Storage.** Store compact views: the front image at model resolution, ~50–100 KB each. About 1M samples then take ~50–100 GB, and the raw data stays referenced, not copied.

### 3c. Reasoner data across sources
- **Decisions:** labels come from rules over each source's recorded future. So every source contributes decision labels for free (D).
- **Teacher text:** costs one teacher call per image, so label a balanced subset across sources.
  - Start with the P5 pilot (~200 samples).
  - Use ReCogDrive's navtrain QA (752k, V) as a comparison arm (D9).
- **Text-rich sources for pretraining:** CoVLA captions, nuReasoning, LingoQA, DriveLM, Nu-X and BDD-X (`DATASETS.md` §1g).

## 4. What the dataset must guarantee [proposed]
| # | Requirement | Why |
|---|---|---|
| R1 | Both models' samples come from the same scenes and tokens | The pilot can be trained on, and later with, the reasoner's outputs |
| R2 | Test-time-visible inputs are stored apart from labels and privileged data (future, boxes, map) | No leaks (`README.md` §2.3) |
| R3 | The full continuous future is kept | Any tokenizer can be fitted later (C2, consumer d) |
| R4 | Token counts are recorded for every rendered view | Compute accounting (C3, consumer h) |
| R5 | The dev split is frozen by log; navtest and navhard never enter | Honest model selection (consumer f) |
| R6 | Every label carries its version | Results trace to exact labels |

## 5. Open choices
These are decided by ablation and do not block P2.
- Front camera only vs the 3-camera stitch; image resolution and visual-token budget.
- Ego history in the pilot prompt, or not: it risks the ego-history shortcut.
- Number format in prompts: decimals, units, special tokens.
- Decision vocabulary and thresholds, calibrated on real distributions in P2.
- Teacher model and prompting style (P5).
- The reasoner's image: one front frame or more cameras.

## 6. Next step: the P2 audit, narrowed to these two models
After your OK (`README.md` §6), run the same checks on a small sample of each shortlisted source: ~10–20 navtrain logs, and a few hundred frames per other source.
- **Conventions:**
  - ego-frame origin and axes;
  - future targets;
  - the command encoding;
  - camera timestamps vs poses;
  - the path-projection check.
- **Distributions:**
  - speed, command, path curvature;
  - decision classes under the v0.1 thresholds (class balance);
  - how often the command is "unknown" or disagrees with the path.
- **Token counts** of candidate pilot and reasoner views.
- **About 50 random samples per source,** rendered as each model would see them, for a visual check.
- **A dev-split proposal** (which logs, city balance) and the dev metric-cache test.
