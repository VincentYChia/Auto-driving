# P1 requirements: data for the two small models (draft, 2026-10-20)

**Status: draft for Phase 1 of `README.md` §4 [proposed].**
- **Scope [agreed 2026-10-20]:** the fast pilot and the reasoner. Multi-frame and object-motion data are deferred.
- **Exit criterion:** you agree with §1–§4, or change them.
- **No code and no field definitions yet.** This fixes *what information* flows and what it looks like, not how it is stored.

Evidence marks (V / R / D) as in `docs/RESEARCH_DIRECTION.md`. The examples are illustrative and invented.

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

## 3. Scale, sources and splits
- **Source:**
  - NAVSIM navtrain only: ~103k samples from 1,192 logs (V).
  - The 8 OpenScene cameras are at 2 Hz (V); only the current frame is used for now.
- **Dev split:**
  - About 5% of navtrain logs, frozen, stratified by city, used for model selection (D2).
  - A dev metric cache for PDMS: feasibility to be checked in P2.
- **Final testing only:** navtest (v1 PDMS) and navhard two-stage (EPDMS). Never train or select on them (V, NAVSIM rule).
- **Teacher text:**
  - First ~200 samples (P5 pilot). Then a subset or all of navtrain, decided from the pilot's cost and quality.
  - ReCogDrive's released navtrain QA (752k QA by Qwen2.5-VL-72B, V) serves as a comparison arm (D9).
- **Optional later** (`DATASETS.md`):
  - human-written examples (Nu-X, BDD-X) as style references for the teacher prompt;
  - nuReasoning, after a log-overlap check.

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
On ~10–20 navtrain logs, after your OK (`README.md` §6):
- **Conventions:**
  - ego poses and future targets;
  - the driving-command encoding;
  - camera timestamps vs poses.
- **Distributions:**
  - speed, command, path curvature;
  - decision classes under the v0.1 thresholds (class balance);
  - how often the command is "unknown" or disagrees with the path.
- **Token counts** of candidate pilot and reasoner views.
- **About 50 random samples** rendered as each model would see them, for a visual check.
- **A dev-split proposal** (which logs, city balance) and the dev metric-cache test.
