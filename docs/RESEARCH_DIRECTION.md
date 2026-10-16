# Research direction (living document)

This document is how we stay aligned between sessions. It records **what we are trying to show and why**.
- `docs/ROADMAP.md` covers the how: plans, compute and order of work.
- `docs/MVLAD_WALKTHROUGH.md` (when written) explains the baseline.

When the direction changes, edit the relevant section and add a dated entry to the **Decision log** at the bottom.

Status markers: **[agreed]** = the researcher has confirmed it · **[proposed]** = a suggestion from research, not yet confirmed · **[open]** = undecided.

---

## 1. The researcher's thought process (as stated, 2026-10-16)

### Target system [agreed]

```
Sensors ──► perception + tracker ──► object-motion tokens ─┐
front image + ego status + driving command ─────────────────┤
                                                             ▼
         FAST LOOP:  VLM quick pilot (Qwen3.5) ──► masked-diffusion action head ──► action
                     (re-plans continuously: produces paths but doesn't execute
                      all of them; keeps refreshing the path)
                                                             │ same inputs
                                                             ▼
         SLOW LOOP:  VLM reasoner (Gemma 4) ──► reasoning (fed into the next fast-loop context)
                                            └─► narration (for interpretability / tracking model ability)
```

Constraint: **the whole stack runs on an in-vehicle edge computer.** This is meant as an edge piloting system, not a cheap server-based alternative.

### Claimed contributions, as originally stated [agreed as a starting point; see §2 for the research check]

Novel:
1. Feed the **past motion of surrounding objects** into the model's context **instead of past images**.
2. Use a **diffusion architecture with more discrete action tokens** than previous work (baseline MVLAD-AD uses 256).
3. Run **entirely on the edge**.

Supporting, not novel:

4. Prompt engineering and quantization. Quantization only if we hit time/latency problems.
5. Fine-tuning.
6. Model architecture choices.

### Next step if the direction holds [agreed]
**Collect the dataset and define its schema.** This step underpins claims 1 and 2, so it has to be done carefully.

---

## 2. Research assessment (2026-10-16) [proposed]

_Filled in from the 2026-10-16 literature and hardware review; see the end of this document._

---

## Decision log

| Date | Decision / change | Status |
|---|---|---|
| 2026-10-16 | Initial direction recorded (system, claims 1–3, supporting 4–6, next step = dataset + schema). Qwen3.5 = fast pilot, Gemma 4 = slow reasoner. Prompt engineering and quantization added as work packages. | agreed |
