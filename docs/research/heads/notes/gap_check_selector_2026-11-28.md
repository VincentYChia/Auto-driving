# Gap check: the selector before and after RL, and in-fold against cross-fitted selector training (2026-11-28)

**Summary**
1. Q1: no driving or robotics paper found reports a learned selector's gain over the generator's own pick before and after RL fine-tuning of the same generator. The closest, DiffusionDriveV2 (arXiv 2512.07745), reports the selector gain only before RL (DiffusionDrive 88.1 -> 89.1 PDMS, Table 9). It trains its selector after RL but reports no row without it, and its RL-tuned generator is less diverse (Div 42.3 -> 30.3, Table 3).
2. Q2: the LLM analogue is established in parts. After RLVR, the base model wins pass@k at large k (Yue et al., 2504.13837, §3.1, §4.4). RL loses sample diversity, and SFT beats RL under reward-model reranking on NaturalQuestions (GopherCite, 2203.11147, §3.5). ReFT's reward-model reranking gain shrinks by 3.8 to 7.4 points after PPO but stays positive (2401.08967, Tables 2 and 4, derived).
3. Q3: no published comparison of in-sample against cross-fitted candidates for training a scorer was found. Jackknifed n-best lists are classic parse-reranking practice (Charniak & Johnson 2005, 20 folds). NAVSIM selectors train on their own generator's proposals without testing that choice (DiffusionDriveV2, FlowR2A, Vault, CLOVER). LEVER's same-generator verifier is best in 5 of 6 cases (2302.08468, Table 8, derived).
4. Where driving papers report a learned scorer's gain after RL, it is small and positive, with no before-RL row on the same policy: +0.2 PDMS on AdaThinkDrive (DriveReward 2606.08525, Table 5) and +0.56 nuPlan score on a PPO policy (DriveZero 2609.06055, Table 2). In LLMs, reranking still helps after RL (ReFT), so our "no gain at all" is stronger than the published pattern.
5. Verdict: finding 1 is New in part (selector gains on imitation-trained generators are published; cross-fitted training on half generators is new). Finding 2 is New in part (no driving version found; the LLM analogue is published). Finding 3 is Open (no published comparison found; navdev only). Findings 2 and 3 cannot be Filled until they are measured on the full navtest.

## How this check was done
- 2026-11-28: 17 web searches (16 standard, 1 extended) and 56 page reads. The pages were arXiv abstract and HTML pages, ar5iv, and one ACL Anthology PDF read by local text extraction. Public papers only.
- Tags: **V** = read in the paper's full text, with the section or table given. **R** = abstract, search snippet or self-report only. **D** = derived by me from the paper's numbers.
- Full texts were read through a page-fetch tool that summarises long pages. Table values are as the tool reproduced them. Where a paper's own numbers disagree with each other, this is noted.
- "Not found" means not found in the searches and reads recorded below (GAPS.md condition 3). It is not proof that no such work exists.
- Our three findings, as given in the brief:
  - **Finding 1.** Without RL, the selector helps: +1.35 on one path generator (navdev), and on the anchored-diffusion path generator +6.02 (navdev) and +3.47 (full navtest).
  - **Finding 2.** After RL on the PDM score, no selector helps. Of 49 selectors on 7 RL-tuned path generators, none has an interval above 0 and 15 are wholly below 0. The candidates concentrate, and the selectors over-predict no-collision.
  - **Finding 3.** After RL, the in-fold selector beats the cross-fitted one: +0.86 (+0.47 to +1.07). Without RL it does not: -0.61 (-1.75 to +1.80).
  - Findings 2 and 3 are measured on navdev only.

## Q1. A learned selector before and after RL on the same generator

**Answer: none found.** No driving (NAVSIM, nuPlan) or robotics paper read here reports a learned scorer's gain over the generator's own pick, on the same generator, both before and after RL fine-tuning. Each paper named in the brief falls into one of three groups:
- no RL on the generator;
- no learned selector;
- DiffusionDriveV2, which reports the selector's gain only before RL.

### Papers named in the brief
| Paper (arXiv) | RL on the generator | Learned selector | Selector gain before vs after RL, same generator | Tag |
|---|---|---|---|---|
| Hydra-MDP (2406.06978) | None mentioned; distilled from human and rule-based teachers | Multi-head decoder that produces candidates for each metric | Not applicable | R (abstract) |
| GTRS (2506.06664) | None mentioned | Scorer over a super-dense vocabulary (with dropout), plus diffusion-generated proposals | Not applicable | R (abstract) |
| DriveSuprim (2506.06659) | None mentioned | Coarse-to-fine selection, rotation augmentation, self-distillation | Not applicable | R (abstract) |
| WoTE (2504.01941) | None mentioned | BEV world model evaluates candidate trajectories online | Not applicable | R (abstract) |
| iPad (2505.15111) | None mentioned | Iterative proposal refinement; selection not described | Not applicable | R (abstract) |
| ZTRS (2510.24108) | No generator: RL (EPO) trains the scorer itself over a fixed vocabulary of 16,384 trajectories (§2.1, §3.2) | The scorer is the policy | Not applicable. Table 4: imitation-trained scorer 86.2 EPDMS, RL-trained 85.3 (benchmark not named) | V |
| WAM-Diff (2512.11872) | GSPO: 86.6 -> 91.0 PDMS at group size 3 (Table 3) | None: one output per scene (§3.4); groups are sampled only during RL training (§3.3) | Not applicable | V |
| WAM-Diff2 (2608.01035) | None (autoregressive-to-diffusion distillation) | None mentioned | Not applicable | R (abstract) |
| ReCogDrive (2506.08052) | DiffGRPO: 86.5 -> 90.8 PDMS (Table 3, latest HTML). A search summary of another version gave 89.6 (+2.8) [R] | None: one denoising chain at inference (§3, §4.2) | Not applicable | V |
| DiffusionDriveV2 (2512.07745) | Intra-anchor and inter-anchor GRPO, started from DiffusionDrive's weights | Two-stage mode selector with BCE + margin-rank loss (§4.6), trained after RL (§5.2, §7) | Before RL only: DiffusionDrive 88.1 -> 89.1 (Table 9). After RL: 91.2 with the selector, no row without it | V |
| ReflectDrive (2509.20109) | None named | None: local search for unsafe tokens, then inpainting | Not applicable | R (abstract) |
| Designing Versatile Samples for Learned Trajectory Scoring (2609.01799, WACV 2027) | None: "Both planners are held fixed throughout the training process" (§5.1) | Transformer scorer on frozen DiffusionDrive and MeanFuser | Not applicable (no RL) | V |

### Other driving papers that combine RL or reward-driven training with a learned selector
- **DiffusionDriveV2 (2512.07745), in detail.**
  - **When the selector is trained:** "The mode selector was trained for 20 epochs using the same configuration" (§5.2). Stage I is RL (10 epochs) and Stage II is the selector (20 epochs) (§7). The selector therefore trains on the RL-tuned generator's own trajectories, with augmentation [V].
  - **Table 9 (navtest):** DiffusionDrive scores 88.1 without the selector and 89.1 with it. DiffusionDriveV2 scores 91.2 with the selector, and there is no row without it [V].
  - **Table 8 (selector design on DiffusionDriveV2):** 89.7, 89.9 and 90.1 for the base single-stage selector, + coarse-to-fine, and + rank loss [V]. Table 8's best row (90.1) is 1.1 below Table 9's 91.2 for the same model with a selector, and no explanation was seen [D].
  - **Where DiffusionDrive's +1.0 comes from:** NC falls 98.2 -> 97.2 and TTC falls 94.7 -> 92.2, while EP rises 82.2 -> 86.8 and DAC rises 96.2 -> 97.0 (Table 9) [V numbers]. This selector buys progress at the cost of collision margin [D]. That is the same failure direction as our selectors' over-prediction of no-collision.
  - **Diversity of the raw trajectories before selection (Table 3):** Div is 42.3 for DiffusionDrive and 30.3 for DiffusionDriveV2, a 28% drop after RL [V numbers, D percentage]. The paper does not discuss the drop. Table 3's PDMS@K ranking is not specified, so it is not used here.
- **DriveReward (2606.08525), §5.4, Table 5.** This is a learned VLM reward model used as a selector on AdaThinkDrive, with N = 4 independent inference passes. The default pick scores 90.3, the DriveReward selector 90.5 (+0.2), and the oracle best-of-4 93.0 (+2.7) [V]. The text gives "3.0%" and "0.22%" and cites the wrong table [V]. AdaThinkDrive's RL training is known here only from its title in the reference list [R]. The paper tests one planner, with no before-RL row and no interval.
- **DriveZero (2609.06055), §2.1.3, Tables 2 and 3.** The policy is trained with PPO.
  - **Candidates and scoring:** the candidates are the modal action plus N-1 sampled first actions. Each is scored by a short rollout plus the PPO critic.
  - **Margin rule:** the best sampled candidate replaces the mode only if it wins by a margin δ, the same rule shape as ours. The paper calls this "a conservative extension of the deployed policy" [V].
  - **Gains:** the nuPlan average rises 93.01 -> 93.57 (+0.56) at N = 64. The gain is +0.11, +0.13 and +0.39 at N = 8, 16 and 32 [V].
  - **Not tested:** no imitation-trained policy is run with the search, and there is no NAVSIM search result [V].
- **CLOVER (2605.15120 v2).** This is not RL. The scorer is refit to evaluator sub-scores on the current generator's proposals, and the generator is refined toward the scorer's top-k and Pareto targets (§3.3) [V].
  - **Table 5 (§5.3), baseline -> Stage 1 -> Stage 2** [V]:

    | Measure | Baseline | Stage 1 | Stage 2 |
    |---|---|---|---|
    | Selected PDMS | 0.9369 | 0.9413 | 0.9448 |
    | Oracle - selected | 0.0564 | 0.0563 | 0.0495 |
    | Oracle@64 | 0.9933 | 0.9976 | 0.9939 |
    | Endpoint area | 5.78 (Table 24) | 46.18 | 25.56 |
    | Clusters@2m | not seen | 10.91 | 8.71 |

    Appendix H (Table 23) gives Stage 2's selected PDMS as 0.9443, not 0.9448.
  - **Reading [D]:** the refinement step concentrates the proposals (endpoint area -45% from Stage 1). The selected result still rises, because the generator improves, but no row gives the generator's own pick without the scorer.
  - **The paper's own warnings:** "directly maximizing such an imperfect score can exploit scorer errors or reduce diversity" (§2.3), and "scalar top-k supervision can over-concentrate the student around a narrow set of high-score modes" (§5.4) [V].
- **Vault (2606.06219 v2).** This is a sample-based reward loop ("without policy gradients, learned reward models, or extra interaction"). A learned scorer picks one of 1,024 candidates.
  - **Table 1:** 94.6 PDMS with the reward-gated pool, 93.9 without it, and an oracle best-of-1,024 of 99.3 [V].
  - **Oracle at each N (§4.3):** without the pool 97.2 / 97.8 / 98.0 at N = 64 / 256 / 1024; with the pool 98.4 / 99.0 / 99.3 [V].
  - **Reading [D]:** here, reward-driven training raised the candidate ceiling rather than shrinking it. The oracle - selected gap is 4.7 with the pool and 4.1 without.
  - **Not reported:** no row without the scorer. The paper says "Whether these terms improve top-one selection requires a separate scorer ablation" [V].
- **AutoVLA (2506.13757), Table 1, §4.2.** One-shot 80.54 PDMS, after GRPO fine-tuning 89.11, and "Best-of-N" 92.12 with "an oracle scorer to select the optimal trajectory from six generated candidates" [V]. The best-of-N row is an oracle, not a learned selector, and its base model is not stated.
- **TrajMoE (2512.07135), §II-C, Table I.** "We adopt the Group Relative Policy Optimization (GRPO) strategy to fine-tune the scoring heads of GTRS", which is RL on the selector, not on a generator [V]. TrajMoE-GRPO scores 42.7 EPDMS against 43.8 for TrajMoE-MoE. The two rows differ in more than RL, so the comparison is not isolated [V].
- **FlowR2A (2606.24231 v2).** No RL. A mode selector "ranks the proposals from the action decoder" (§3.2), and a second training stage fine-tunes it on the decoder's own proposals (§3.3; see Q3) [V].
- **DIVER (2507.04049).** GRPO with diversity and safety rewards. The final trajectory is "the trajectory with the highest score", following SparseDrive and DiffusionDrive (§IV-B3). Whether that score is learned is not described [V]. Div rises 0.21 -> 0.35 on Bench2Drive against the SparseDrive baseline when both rewards are added (Table X) [V]. So an explicit diversity reward can raise diversity, but no selector result is given before or after RL.
- **IRL-VLA (2508.06571)** uses a learned reward world model as the PPO reward. **DriveDPO (2509.17940, NeurIPS 2025)** distils a policy from imitation and rule-based safety scores, then applies Safety DPO. Neither reports selection results in the abstracts and snippets read [R].
- **PlannerRFT (CVPR 2026)** and **Plan-R1 (ICLR 2026)** were seen as search snippets only. PlannerRFT: a "lack of diversity in candidate groups fails to provide effective optimization signals for reinforcement fine-tuning" [R]. This concerns the RL signal, not a selector.

### Robotics
- **PA-RL** ("Policy Agnostic RL", 2412.06685), Table 4 (§5.3). The ablation compares "no global opt." (no Q-function re-ranking of actions sampled from the policy) with full PA-RL, offline -> after online fine-tuning [V]:
  - CALVIN: 215 -> 389 without re-ranking, 234 -> 455 with it.
  - antmaze-large-diverse: 0 -> 0 without, 73 -> 93 with.

  On CALVIN the re-ranking step's contribution grows from +19 to +66 after online fine-tuning [D], the opposite direction to our finding 2. It is not like-for-like: "no global opt." is a separately trained variant (re-ranking also shapes the distillation targets), and the critic is retrained online on the policy's own data.
- **V-GPS** ("Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance", 2410.13816, CoRL 2024). An offline-RL value function re-ranks actions from frozen generalist policies. "The same value function can improve the performance of five different state-of-the-art policies with different architectures" [R]. None of those policies is RL-tuned.
- **RoboMonkey** (2506.17811). A VLM-based action verifier over sampled actions: +25% absolute out of distribution and +9% in distribution. Fine-tuning both the VLA and the verifier gives +7% over fine-tuning the VLA alone on LIBERO-Long [R, snippet]. No RL-tuned policy is tested.
- Query 5 found no robotics paper that tests a selector on one policy before and after RL fine-tuning.

### What Q1 supports [D]
- **The only same-paper material is DiffusionDriveV2's.** The selector adds +1.0 on the imitation-trained generator, and after RL the raw trajectories are 28% less diverse. The selector's gain after RL is not reported.
- **Gains after RL are small; gains before RL are larger.**
  - After RL, the reported learned-scorer gains are +0.2 PDMS (DriveReward, N = 4) and +0.11 to +0.56 nuPlan score (DriveZero, N = 8 to 64).
  - Before RL, the reported gains on imitation-trained generators are +1.0 PDMS (DiffusionDriveV2, Table 9) and +1.4 to +1.8 EPDMS (Versatile Samples, §5.2).

  The generators, selectors and metrics differ, so this is a cross-paper pattern in the direction of our finding 2, not a controlled comparison. Our zero-to-negative result is stronger than these small positives.
- **The positive after-RL cases use selectors trained on the RL-tuned policy's own data:** DriveZero's PPO critic and DiffusionDriveV2's Stage II. That fits our finding 3 (in-fold is better after RL).
- **Oracle headroom remains after reward-driven training:** AutoVLA best-of-6 +3.0 over its post-GRPO pick (base model of that row unstated), DriveReward best-of-4 +2.7, Vault best-of-1,024 +4.7 over its selected path. In published systems, the learned scorers recover little of this headroom.

## Q2. The LLM analogue: RL narrows sampling, so best-of-N and rerankers gain less

**Answer: established for the sampling side, and only partly for learned rerankers.** RL fine-tuning (RLVR/GRPO/PPO, RLHF) narrows the sample distribution and lowers large-N coverage. Learned-reranker gains shrink after RL but usually stay positive.

### RL narrows the sample distribution and the best-of-N / pass@k headroom
- **Yue et al.**, "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", arXiv 2504.13837 (NeurIPS 2025 oral) [V]:
  - RLVR models beat their base models at small k, but "base models consistently catch up to and eventually surpass RL-trained models across all benchmarks" (§3.1, Fig. 2). On Minerva at 32B, the base model is about 9% higher at k = 128.
  - "As RL training progresses, pass@256 progressively decreases, indicating a reduced reasoning boundary", while training pass@1 rises from 26.1 to 42.5 (§4.4).
  - Output entropy typically falls during RL, which contributes to the narrowing but does not fully explain it (§4.5).
  - For Magistral-Medium, RLVR gives "significant gains at low k, but little or no improvement at higher k" (§4.6, Fig. 9).
- **Wu et al.**, "The Invisible Leash: Why RLVR May or May Not Escape Its Origin", arXiv 2507.14843: "the shrinkage of observed support outweighs its expansion", and RLVR "can make existing successes more reliable while losing access to other verified solutions" [R].
- **He, Fried & Welleck**, "Rewarding the Unlikely: Lifting GRPO Beyond Distribution Sharpening", arXiv 2506.02355. GRPO's rank bias sharpens the distribution: the tuned model "can solve some problems with fewer samples, but underperforms simply sampling more solutions from the original model". An unlikeliness reward improves pass@N over a large range of N in theorem proving [R].
- **Kirk et al.**, "Understanding the Effects of RLHF on LLM Generalisation and Diversity", arXiv 2310.06452: "RLHF significantly reduces output diversity compared to SFT across a variety of measures", while generalising better [R].
- **Matsutani et al.**, "RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs", arXiv 2509.21128 (ICLR 2026): "RL compresses incorrect trajectories, whereas SFT expands correct ones" [R].
- **Cobbe et al.**, "Training Verifiers to Solve Math Word Problems", arXiv 2110.14168. This is the same mechanism under supervised training, not RL [V]:
  - "test@100 performance degrades much more sharply than test@1 as we increase the number of epochs" (§4.1, Fig. 3).
  - The generator that feeds the verifier is stopped at 2 epochs: "We choose not to train for longer, since the diversity of generated solutions begins to collapse after this point" (§4.2).
- **Wang & Ying**, "Do Visual RL Gains Scale at Test Time? A Quantile View of Post-Training", arXiv 2611.02366, on image generation [R]:
  - Under best-of-N, RL gains may "attenuate, saturate, or even reverse at large sampling budgets".
  - RL "can improve typical samples while compressing or degrading the rare high-quality outcomes that test-time scaling exploits".
  - The abstract does not say whether best-of-N uses an oracle or a learned scorer.

### Learned rerankers and verifiers after RL
- **Menick et al.**, "Teaching language models to support answers with verified quotes" (GopherCite), arXiv 2203.11147, §3.5, Tables 1 and 2, Figs. 5 and 6 [V]:
  - "Over the course of RL finetuning we lose the diversity of sampled answers", which "diminishes the benefit of drawing a large number of samples to pick the best one from".
  - On NaturalQuestionsFiltered, "In the reranking regime, RL is outperformed by SFT". The top@64 Supported & Plausible rate is 80.0 +/- 6.1 for SFT and 74.8 +/- 6.7 for RL (Table 1a).
  - On ELI5, "RL outperforms SFT consistently for all numbers of candidates".
  - The RL policy starts from the SFT model (280B Gopher), and the reranker is a 7B reward model.
- **Luong et al.**, "ReFT: Reasoning with Reinforced Fine-Tuning", arXiv 2401.08967 (ACL 2024), using PPO. GSM8K accuracy, single answer (Table 2) and with reward-model reranking (Table 4, §4.4) [V]:

  | Model, prompt | SFT | ReFT | SFT + reranking | ReFT + reranking |
  |---|---|---|---|---|
  | Galactica, N-CoT | 42.68 | 48.14 | 57.5 | 59.2 |
  | Galactica, P-CoT | 58.83 | 68.91 | 73.4 | 76.4 |
  | CodeLLAMA, N-CoT | 43.59 | 53.30 | 62.9 | 66.0 |
  | CodeLLAMA, P-CoT | 63.68 | 75.28 | 77.0 | 81.2 |

  - **Derived [D]:** the reranking gain shrinks from before RL to after RL in all four rows: +14.8 -> +11.1, +14.6 -> +7.5, +19.3 -> +12.7 and +13.3 -> +5.9. That is a drop of 3.8 to 7.4 points, and the gain stays positive. ReFT + reranking still beats SFT + reranking ("by more than 3 points").
  - **Majority voting shrinks less [D]:** its N-CoT gain is unchanged at about +10, and its P-CoT gain goes from +4.1 / +4.3 to +2.9 / +2.7.
  - **Caveats [V]:** the reward model is trained on samples from the warm-up checkpoint, not from either final policy (§4.3). Table 4 lists CodeLLAMA as 0.7B, while §4.3 says 7B.
- **Feng et al.**, "Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs", arXiv 2505.11227 (NeurIPS 2025). On strong RL-trained models (DeepSeek-R1, QwQ-32B), "current PRMs underperform simple baselines like majority voting". The model's own judgement (Self-PRM) does improve accuracy [R]. These are external PRMs, not a before/after comparison on one model.
- **Chow et al.**, "Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models", arXiv 2412.15287. This is a remedy, not a measurement of the loss. Training for the best-of-N objective makes the model interleave best responses with "more diverse responses". Gemma 2B Bo32 on MATH rises 26.8% -> 30.8%, and pass@32 rises 60.0% -> 67.0% [R].

### Counter-evidence
- **Liu et al.**, "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models", arXiv 2505.24864. With KL control, reference-policy resets and a diverse task suite, "RL-trained models consistently outperform base models across a wide range of pass@k evaluations". The effect "correlates strongly with task competence of base model and training duration" [R].
- **Wu et al.**, "Position: The Hidden Costs and Measurement Gaps of Reinforcement Learning with Verifiable Rewards", arXiv 2509.21882: several widely cited RLVR gaps "shrink substantially or disappear once budgets, prompts, and dataset versions are matched" [R]. This is a caution for both sides of the pass@k debate.
- In ReFT, and in GopherCite on ELI5, reranking still adds after RL [V].

### What Q2 supports [D]
- **Established:** RL narrows the sample distribution, and at large N the base model's coverage exceeds the RL model's. Over-training also collapses diversity under supervised training (Cobbe).
- **Partly established for learned rerankers:** their gains shrink after RL (ReFT, derived; GopherCite on NaturalQuestions), and external PRMs fall below majority voting on strong RL models. But positive reranking gains after RL are also published (ReFT; GopherCite on ELI5).
- **Not found:** no LLM paper read here shows a reranker's gain going to zero or below, on the same model after RL, with intervals.

## Q3. Training the scorer on in-sample against cross-fitted candidates

**Answer: no published comparison found**, in driving, robotics or LLM verifier and reranker training.

### Cross-fitting (jackknifing) as established practice
- **Charniak & Johnson**, "Coarse-to-Fine n-Best Parsing and MaxEnt Discriminative Reranking", ACL 2005, P05-1022 [V; PDF read by local text extraction]:
  - "the cross-validation scheme described in Collins (2000) was used to avoid training the n-best parser on the sentence it was being used to parse" (§1).
  - "We used the 20-fold cross-validation technique described in Collins (2000) to compute the n-best parses Y(s) for each sentence s in D" (§4).
  - No comparison with in-sample n-best lists appears in the text read.
- Collins (2000), the cited origin of the scheme, was not read here.

### LLM verifier training
- **Cobbe et al.** (2110.14168). The verifier trains on 100 completions per training problem, sampled from the generator that was fine-tuned on those same problems (§4.2), so the candidates are in-sample. In-sample optimism is handled by stopping the generator at 2 epochs, not by folds. Cross-validation and held-out generators are not discussed [V].
- **Ni et al.**, "LEVER: Learning to Verify Language-to-Code Generation with Execution", arXiv 2302.08468, Appendix B.1, Table 8. Verifiers trained on one code LLM's samples are applied to another's. Each cell is the target's score with the verifier trained on that column's LLM; the target's own verifier is listed first in each row [V]:

  | Dataset, target (baseline) | Own verifier | Codex-trained | InCoder-trained | CodeGen-trained |
  |---|---|---|---|---|
  | Spider, Codex (77.3) | 82.0 | own | 81.7 | 80.8 |
  | Spider, InCoder (41.2) | 54.1 | 46.4 | own | 47.6 |
  | Spider, CodeGen (37.7) | 51.0 | 44.7 | 48.9 | own |
  | GSM8k, Codex (72.1) | 83.7 | own | 70.0 | 71.9 |
  | GSM8k, InCoder (4.3) | 11.9 | 8.3 | own | 12.3 |
  | GSM8k, CodeGen (9.6) | 22.1 | 18.4 | 20.7 | own |

  - **Derived [D]:** the same-generator verifier is best in 5 of 6 rows. The cross-generator loss is largest for the strongest generator: Codex on GSM8k gets +11.6 from its own verifier and -2.1 and -0.2 from the others.
  - **From the paper:** transfer works better when the positive-label rates are closer. The samples are few-shot (no fine-tuned generator), so this measures distribution match, not in-sample optimism [V].
- **ReFT** (2401.08967, §4.3): the reranker is trained on samples from the warm-up checkpoint and applied to both the SFT and the ReFT policy. It still helps both (see Q2) [V].
- **Ackermann, Ishida & Sugiyama**, "Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback", arXiv 2507.15507 (COLM 2025). As RL proceeds, "the responses generated by the LM no longer resemble the responses seen by the RM during training", "leading to the RM becoming inaccurate". The fix is importance weighting [R].
- **Lang, Huang & Li**, "Fine-Tuning Language Models with Reward Learning on Policy", arXiv 2403.19279 (NAACL 2024). Fixed reward models become inaccurate off-distribution, and RLP "refines a reward model using policy samples to keep it on-distribution" [R].

### Driving selectors: all train in-fold, and none compares
- **DiffusionDriveV2:** the selector is trained in Stage II on the RL-tuned generator's trajectories, with augmentation (§5.2, §7) [V].
- **FlowR2A** (2606.24231, §3.3) [V]:
  - The stage-2 selector is fine-tuned on "online proposals labeled via simulation" because "The stage-1 selector sees only vocabulary trajectories, which differ from decoder proposals at inference".
  - Random vocabulary trajectories are mixed in "to keep the selector calibrated across the full quality range".
  - No ablation isolates stage 2, and the paper does not say which scenes the stage-2 proposals come from.
- **Vault** (2606.06219, §4.1): "Training labels come from the official scorer applied to the generated trajectories", and the candidates are regenerated on training scenes every T iterations. There is no comparison with other proposal sources [V].
- **CLOVER** (2605.15120): the scorer is refit on the current generator's proposals (§3.3) [V].
  - Its fixed-proposal diagnostic (Appendix F, Table 20) scores the scorer's selected top-1 on navtrain and on navtest. Train is above test for every feature backbone: 95.0 vs 92.4 (DINOv2-S) up to 97.7 vs 92.8 (Wan2.2-5B) [V].
  - That train-test gap is the in-sample optimism that jackknifing targets [D].
- **Versatile Samples** (2609.01799): the training pool "carries designed samples alongside the planner's own candidates" (Fig. 2) [V]. A search snippet says the checkpoint is chosen on a held-out ~5% of navtrain [R]. There is no fold-based candidate generation.
- Query 16 found no NAVSIM work that cross-fits candidate generation for a scorer [R].

### What Q3 supports [D]
- **The comparison is not answered in print.** Two published lines point in opposite directions:
  - In-sample optimism argues for cross-fitting: jackknifing (Charniak & Johnson), Cobbe's 2-epoch cap, and CLOVER's train-test gap.
  - Distribution match argues for training on the deployed generator's own candidates: LEVER's same-generator advantage, OCRM, RLP, and FlowR2A's stage 2.
- **Our pattern fits the second line once RL has run.** Without RL, cross-fitting is not worse; after RL, in-fold is better. That would fit distribution match dominating once RL has moved the deployed generator away from the half generators, each RL-tuned on half of the drives. This mechanism is untested, both by us and in the papers read.

## Queries run
| # | Mode | Query | Outcome |
|---|---|---|---|
| 1 | standard | DiffusionDriveV2 reinforcement learning truncated diffusion GRPO NAVSIM arXiv | Found 2512.07745; HTML read twice (selector trained after RL; Tables 3, 8, 9). |
| 2 | standard | WAM-Diff masked diffusion VLA autonomous driving reinforcement learning trajectory selection NAVSIM | Found WAM-Diff 2512.11872 and WAM-Diff2 2608.01035 (also WAM-Flow, DriveFine); neither reports selection before vs after RL. |
| 3 | standard | NAVSIM learned trajectory scorer after GRPO fine-tuning of planner selector gain diminishes RL candidates | Leads: CLOVER 2605.15120, TrajMoE 2512.07135, FlowR2A 2606.24231, DriveReward 2606.08525, Versatile Samples, ZTRS. No before/after selector result. |
| 4 | standard | end-to-end driving RL fine-tuned generator plus learned trajectory scorer ablation "with and without RL" selector NAVSIM 2026 | Leads: Vault 2606.06219, DriveZero 2609.06055, EvaDrive 2508.09158, MindDrive 2512.04441, ReCogDrive. None has the comparison. |
| 5 | standard | robot VLA test-time action sampling verifier or value reranking gain on RL fine-tuned policy versus base policy (V-GPS, RoboMonkey) | RoboMonkey 2506.17811, V-GPS 2410.13816, a verifier survey 2609.09250. No RL-tuned policy tested. |
| 6 | standard | best-of-N reranking with reward model gains smaller on RL-tuned policy than SFT policy reduced diversity verifier gain after RLHF | GopherCite 2203.11147 (key), Kirk et al. 2310.06452, ReFT 2401.08967, Chow et al. 2412.15287. Course project reports surfaced, not used. |
| 7 | standard | RLVR trained model best-of-N with verifier or reward model gains shrink compared to base model test-time scaling diminished after reinforcement learning 2025 | Yue et al. 2504.13837, Invisible Leash 2507.14843, Position paper 2509.21882, visual-RL quantile paper 2611.02366. No learned-verifier before/after study. |
| 8 | standard | Charniak Johnson 2005 n-best parsing discriminative reranking 20-fold cross-validation jackknife training data n-best lists | ACL P05-1022. PDF text extracted locally: 20-fold cross-validation confirmed (§1, §4). |
| 9 | standard | reranker training n-best lists jackknifing versus no jackknifing comparison base model trained on same data overly optimistic candidates | No paper comparing jackknifed with in-sample n-best training found. |
| 10 | standard | verifier or reward model trained on the policy's own samples versus samples from other models, best-of-N reranking accuracy, on-policy data distribution shift study | OCRM (reward-model drift as the policy moves) and on-policy vs off-policy preference-data papers. No best-of-N comparison. |
| 11 | standard | Off-Policy Corrected Reward Modeling for Reinforcement Learning from Human Feedback Ackermann arXiv | OCRM = 2507.15507 (COLM 2025); also RLP 2403.19279. |
| 12 | standard | train verifier on samples from generators fine-tuned on held-out folds (k-fold) to avoid memorized training problems, in-sample generator samples too correct, math or code reranking | Nothing direct. Cobbe et al. 2110.14168 surfaced (read). |
| 13 | extended | autonomous driving planner GRPO or PPO fine-tuning reduces trajectory candidate diversity, learned scorer or selector gives less improvement after RL, NAVSIM nuPlan ablation | DIVER 2507.04049, AutoVLA 2506.13757, PlannerRFT (CVPR 2026), Plan-R1 (ICLR 2026). Diversity concerns in RL; no selector before/after. |
| 14 | standard | process reward model best-of-N improvement smaller for RL-trained reasoning model than for SFT or base model, PRM reranking after GRPO diminishing returns | Is PRM Necessary? 2505.11227 and RL Squeezes, SFT Expands 2509.21128. No controlled before/after. |
| 15 | standard | "out-of-fold" OR "cross-fitted" candidates to train reranker or verifier for LLM generations, compared with in-sample candidates from the same fine-tuned generator | Nothing direct (V-STaR 2402.06457 and Weaver 2506.18203 surfaced). |
| 16 | standard | trajectory scorer trained on proposals from planners trained on held-out folds (cross-fitting, k-fold, jackknife) to avoid in-sample optimism, end-to-end driving NAVSIM selector training data | No NAVSIM cross-fitting found (GTRS, DriveSuprim, CLOVER, Versatile Samples surfaced). |
| 17 | standard | DriveDPO Safety DPO end-to-end autonomous driving policy distillation trajectory vocabulary NAVSIM arXiv | DriveDPO 2509.17940 (NeurIPS 2025). No selector before/after in the snippets. |

### Pages read (in order)
1. arxiv.org/abs/2609.01799 (Versatile Samples, abstract).
2. arxiv.org/html/2512.07745 (DiffusionDriveV2: selector timing, Tables 3, 8, 9).
3. arxiv.org/html/2512.07745 (DiffusionDriveV2: Tables 3, 8, 9 reproduced).
4. arxiv.org/abs/2605.15120 (CLOVER, abstract).
5. arxiv.org/abs/2606.08525 (DriveReward, abstract).
6. arxiv.org/abs/2606.24231 (FlowR2A, abstract).
7. arxiv.org/abs/2608.01035 (WAM-Diff2, abstract).
8. arxiv.org/html/2605.15120v2 (CLOVER: §3.3, Table 5, §5.4).
9. arxiv.org/html/2512.11872v1 (WAM-Diff: §3.3, §3.4, Table 3).
10. arxiv.org/html/2512.07135v1 (TrajMoE: §II-C, Table I).
11. arxiv.org/html/2605.15120v2, second half (CLOVER: Appendices F and H).
12. arxiv.org/abs/2506.08052 (ReCogDrive, abstract).
13. arxiv.org/abs/2509.20109 (ReflectDrive, abstract).
14. arxiv.org/abs/2510.24108 (ZTRS, abstract).
15. arxiv.org/abs/2506.06664 (GTRS, abstract).
16. arxiv.org/abs/2606.06219 (Vault, abstract).
17. arxiv.org/abs/2609.06055 (DriveZero, abstract).
18. arxiv.org/html/2606.06219v2 (Vault: Table 1, §4.1, §4.3).
19. arxiv.org/abs/2609.09250 (robot verifier survey, abstract). Does not address the questions; not read further.
20. arxiv.org/abs/2504.13837 (Yue et al., abstract).
21. ar5iv 2203.11147 (GopherCite: §3.5, Tables 1 and 2).
22. arxiv.org/abs/2401.08967 (ReFT, abstract).
23. arxiv.org/html/2401.08967v3 (ReFT: Table 4, §4.3).
24. arxiv.org/html/2401.08967v3 (ReFT: Table 2).
25. arxiv.org/abs/2611.02366 (visual-RL quantile paper, abstract).
26. arxiv.org/abs/2507.14843 (Invisible Leash, abstract).
27. arxiv.org/abs/2509.21882 (Position paper, abstract).
28. arxiv.org/abs/2310.06452 (Kirk et al., abstract).
29. ar5iv 2110.14168 (Cobbe et al.: §4.1, §4.2, Fig. 3).
30. aclanthology.org/P05-1022.pdf (served as binary; text extracted locally, §1 and §4).
31. arxiv.org/html/2606.24231v2 (FlowR2A: §3.2, §3.3, §4.3).
32. johannesack.github.io OCRM PDF (HTTP 404).
33. arxiv.org/abs/2507.15507 (OCRM, abstract).
34. arxiv.org/abs/2403.19279 (RLP, abstract).
35. arxiv.org/html/2507.04049 (DIVER: §IV-B3, Table X).
36. arxiv.org/html/2506.13757 (AutoVLA: Table 1, §4.2).
37. arxiv.org/html/2412.06685 (PA-RL: Table 4, Appendix B.1).
38. arxiv.org/abs/2506.02355 (Rewarding the Unlikely, abstract).
39. arxiv.org/abs/2505.24864 (ProRL, abstract).
40. arxiv.org/pdf/2307.09288 (Llama 2). Failed: over the size limit; not used.
41. arxiv.org/abs/2406.06978 (Hydra-MDP, abstract).
42. arxiv.org/abs/2506.06659 (DriveSuprim, abstract).
43. arxiv.org/abs/2504.01941 (WoTE, abstract).
44. arxiv.org/abs/2505.15111 (iPad, abstract).
45. ar5iv 2302.08468 (LEVER: Appendix B.1, Table 8).
46. arxiv.org/abs/2412.15287 (Chow et al., abstract).
47. arxiv.org/abs/2508.06571 (IRL-VLA, abstract).
48. arxiv.org/abs/2505.11227 (Is PRM Necessary?, abstract).
49. arxiv.org/abs/2509.21128 (RL Squeezes, SFT Expands, abstract).
50. arxiv.org/html/2609.01799 (Versatile Samples: §1, §3.2, §5.1, §5.2, Tables 1, 2, 4, 5).
51. arxiv.org/abs/2410.13816 (V-GPS, abstract).
52. arxiv.org/html/2510.24108v1 (ZTRS: §2.1, §3.2, Table 4).
53. arxiv.org/html/2506.08052 (ReCogDrive: §3, §4.2, Table 3).
54. arxiv.org/html/2504.13837v5 (Yue et al.: §3.1, §4.4, §4.5, §4.6).
55. arxiv.org/html/2609.06055 (DriveZero: §2.1.3, Tables 2 and 3).
56. arxiv.org/html/2606.08525 (DriveReward: §5.4, Table 5).

Leads not followed within the budget:
- Collins (2000), the origin of the 20-fold scheme;
- the full texts of PlannerRFT, Plan-R1, EvaDrive, MindDrive, RoboMonkey and V-STaR;
- Llama 2's rejection-sampling temperature observation (its PDF could not be fetched, so it is not used).

## Verdict for the gap list
GAPS.md's rule for Filled needs three things: our own full-navtest measurement, a controlled comparison, and no published version of the same comparison found. Findings 2 and 3 are measured on navdev only, so neither can be Filled now.

**Finding 1: without RL, the selector helps.**
- **Published version: yes.**
- **Closest work:**
  - DiffusionDriveV2 (2512.07745), Table 9: DiffusionDrive scores 88.1 PDMS without its mode selector and 89.1 with it, on navtest [V].
  - Designing Versatile Samples (2609.01799), §5.2: "The gain over the base planner is large: up to 1.8 EPDMS on DiffusionDrive and the plain scorer accounts for 1.4" [V].
- **Suggested: New in part.** The selector-versus-own-pick comparison on an imitation-trained generator is published, so condition 3 fails.
- **What is new:**
  1. The selector is trained on cross-fitted candidates, from half generators that went through the same RL stage as the deployed generator. No NAVSIM precedent was found; the NLP precedent is jackknifed n-best lists (Charniak & Johnson 2005).
  2. The path generators read the hidden states of a frozen 0.8B VLM.
  3. The gain on the anchored-diffusion path generator is larger: +3.47 on navtest, against +1.0 for DiffusionDrive in DiffusionDriveV2's Table 9. The generators, selectors and K differ, so the two are not like-for-like [D].

**Finding 2: after RL, the selector stops helping.**
- **Published version: partly.**
  - **In driving, none found.** No paper reports selector-versus-own-pick on the same generator before and after RL. The closest, DiffusionDriveV2, gives only the before-RL row. After RL it reports "The mode selector was trained for 20 epochs using the same configuration" but has no row without the selector, and its RL-tuned generator's Div falls 42.3 -> 30.3 [V].
  - **The LLM analogue is published.** GopherCite §3.5: "Over the course of RL finetuning we lose the diversity of sampled answers ... In the reranking regime, RL is outperformed by SFT" [V]. ReFT: the reranking gain shrinks by 3.8 to 7.4 points after PPO [D]. Yue et al. §3.1 and §4.4 [V]. Is PRM Necessary? [R].
- **Suggested: New in part.**
- **What is new:**
  1. It is the first measurement in driving of a learned selector's gain over the own pick on the same generator, before and after RL on the PDM score (Dr. GRPO, with and without a BC term).
  2. The gain goes to zero or below, rather than shrinking: across 7 RL-tuned path generators x 7 selector variants, none of the 49 selectors has an interval above 0 and 15 are wholly below. In the published after-RL cases the gain stays positive: ReFT reranking +5.9 to +12.7 accuracy points, DriveReward +0.2 PDMS, DriveZero +0.56 nuPlan score [D].
  3. The mechanism measurements are new: the candidates' end points lie within about 0.7 m of each other in three quarters of scenes, and the selectors over-predict no-collision. No published measurement of either was found. Versatile Samples only notes that imitation-trained proposals "overlap heavily and are mostly collision-free", with no calibration analysis.
- **Partial tension, not a conflict:** reranking still helps after RL in ReFT and in GopherCite on ELI5. PA-RL's critic selection matters more after online fine-tuning on CALVIN. Vault's reward loop raised its best-of-1,024 ceiling. None of these is a NAVSIM selector on a PDMS-RL-tuned generator.
- **On full navtest:** the driving comparison would still have no published version found, but the published LLM analogue keeps New in part as the accurate label.

**Finding 3: after RL, the in-fold selector beats the cross-fitted one.**
- **Published version: none found** for the controlled comparison.
- **Closest work:**
  - Charniak & Johnson 2005, §4: "We used the 20-fold cross-validation technique described in Collins (2000) to compute the n-best parses". This is the practice, with no comparison [V].
  - LEVER (2302.08468), Table 8: the same-generator verifier is best in 5 of 6 rows [D from V].
  - FlowR2A §3.3: "The stage-1 selector sees only vocabulary trajectories, which differ from decoder proposals at inference" [V].
  - OCRM (2507.15507): the reward model becomes inaccurate as the policy moves [R].
- **Suggested: Open.** Condition 3 holds as far as these searches go, but condition 1 fails because the result is on navdev only. On a full-navtest measurement it would become a Filled candidate.
- **What a write-up could claim then:** the sign of in-fold minus cross-fitted flips with RL. It is -0.61 without RL (interval includes 0) and +0.86 after RL (pooled over 7 RL-tuned path generators). That runs against the jackknifing rationale of in-sample optimism, and it fits the distribution-match evidence (LEVER, OCRM, RLP) [D]. The mechanism is untested.
