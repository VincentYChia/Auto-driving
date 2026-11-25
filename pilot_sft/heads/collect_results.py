"""Collect every scored path-generator configuration into one aggregate JSON for reports (no per-scene data).

    python pilot_sft/heads/collect_results.py --out <json>

Sections:
- ``references``: H1's text output and constant velocity, on navdev and navtest.
- ``runs``: every training run with navdev choices (``choices_navdev.json``), at those choices. Single path and
  best-of-6 on navdev (where choices are made, R1) and navtest (report only), each with:
  - a 95% drive interval (R3) and the paired difference to H1's text output;
  - sub-score means and best-of-6 diversity;
  - dev ADE at the chosen epoch.
- ``layer_tests``: per method, layers 12/18/24 minus layer 24, each run at its own choices, paired on the same
  scenes.
- ``comparisons``: every pair of methods at their layer choices, paired on the same scenes.
- ``rl``: every RL run (the R12 recipe runs, phase 4). On navdev, at update 0 (the start) and each saved
  checkpoint:
  - single path and best-of-6, with the paired change from the start;
  - sub-score means and diversity;
  - plus the training curve in blocks of 50 updates.
- ``series``: phase 2 (R17): A's continuity series, leave-one-out and bin-size runs, each minus A1 at its own
  choices, paired, with R17's test (more than 1.4 points and a drive interval excluding 0).
- ``selectors``: phase 6 (R14): every trained selector's held-out choice, training, and R14's measures on navdev
  (and navtest once run).
- ``selector_candidates``: per chain, the headroom of each completely scored candidate set (training scenes).
- ``selector_halves``: per chain, its half generators against the deployed one on navdev (R14's measure).
- ``selector_oof_pairs``: every pair of chains on the same out-of-fold training scenes (``oof_pair.py``).
- ``methods``: the choice files the driver wrote (layer choices, RL starts, the R12 recipe, decisions).

Intervals resample drives: 10,000 resamples, seed 0. Runs on the desktop with the score CSVs (pilot env).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import diversity, read_paths  # noqa: E402
from candidate_headroom import headroom  # noqa: E402

HOME = os.path.expanduser("~")
SPLITS = ("navdev", "navtest")
REF = {"navdev": "h1-text_navdev_v1", "navtest": "deskcheck-h1_navtest_v1"}
CV = {"navdev": "cv_navdev_v1", "navtest": "cv_navtest_v1"}
SUBS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "ego_progress",
        "time_to_collision_within_bound", "comfort")
SELECTOR_KEYS = ("scenes", "candidates", "drives", "selected_pdms", "own_pick_pdms", "selected_minus_own",
                 "selected_minus_own_ci", "best_of_k_pdms", "random_pick_pdms", "mean_candidate_pdms",
                 "share_of_best_of_k_gap_closed", "switch_rate", "switches", "picks_failing_nc_or_dac", "top1_exact",
                 "top1_tie_aware", "gain_by_spread_quartile", "calibration")  # R14's measures, from selector.report
R17_THRESHOLD = 0.014  # 1.4 PDMS points: two seed SDs of a single-seed difference (docs/HEADS_PLAN.md R17)
CURVE = ("reward", "best_of_g", "equal_reward_groups", "kl", "grad_norm", "no_at_fault_collisions",
         "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress", "paths_per_s", "bc")


def drive_ci(x: np.ndarray, groups: np.ndarray, n: int = 10000) -> List[float]:
    u, inv = np.unique(groups, return_inverse=True)
    s, c = np.bincount(inv, weights=x), np.bincount(inv)
    i = np.random.default_rng(0).integers(0, len(u), size=(n, len(u)))
    m = s[i].sum(1) / c[i].sum(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


class Results:
    """Score CSVs and path folders under ``exp``; drive maps from ``cache_dir``."""

    def __init__(self, exp: str, cache_dir: str):
        self.exp, self.heads = exp, os.path.join(exp, "heads")
        self.drives = {sp: json.load(open(os.path.join(cache_dir, f"drives_{sp}.json"))) for sp in SPLITS}
        self._frames: Dict[str, Optional[pd.DataFrame]] = {}
        self._at: Dict[str, Dict[str, dict]] = {}
        self.refs = {sp: self.score_of(REF[sp]) for sp in SPLITS}

    # ---- score files ----------------------------------------------------------------------------------------
    def frame(self, name: str) -> Optional[pd.DataFrame]:
        if name not in self._frames:
            f = sorted(glob.glob(os.path.join(self.exp, name, "*", "*.csv")))
            d = None
            if f:
                d = pd.read_csv(f[-1])
                d = d[d.token != "average"].set_index("token")
            self._frames[name] = d
        return self._frames[name]

    def score_of(self, name: str) -> Optional[pd.Series]:
        d = self.frame(name)
        return None if d is None else d["score"].astype(float)

    @staticmethod
    def name(run: str, sp: str, weights: str, steps: int, folder: str) -> str:
        return f"heads-{run}-{sp}_{weights}-steps{steps}-{folder}_{sp}_v1"

    def folder(self, run: str, sp: str, weights: str, steps: int, folder: str) -> str:
        return os.path.join(self.heads, run, f"{sp}_{weights}", f"steps{steps}", folder)

    @staticmethod
    def sample_names(scale) -> List[str]:
        return [f"sample_{j}" if scale == 1.0 else f"scale{scale:g}_sample_{j}" for j in range(6)]

    # ---- statistics -----------------------------------------------------------------------------------------
    def stats(self, s: pd.Series, sp: str) -> dict:
        g = np.array([self.drives[sp][t] for t in s.index])
        out = {"pdms": round(float(s.mean()), 4), "ci_drives": drive_ci(s.values, g), "scenes": int(len(s)),
               "drives": int(len(set(g)))}
        if self.refs[sp] is not None:
            out["vs_h1_text"] = self.paired(s, self.refs[sp], sp)
        return out

    def paired(self, a: pd.Series, b: pd.Series, sp: str) -> dict:
        common = a.index.intersection(b.index)
        d = a.loc[common].values - b.loc[common].values
        return {"difference": round(float(d.mean()), 4),
                "ci_drives": drive_ci(d, np.array([self.drives[sp][t] for t in common])), "scenes": int(len(common))}

    def subscores(self, d: pd.DataFrame) -> dict:
        return {k: round(float(d[k].astype(float).mean()), 4) for k in SUBS if k in d.columns}

    # ---- one configuration ----------------------------------------------------------------------------------
    def single(self, run: str, sp: str, weights: str, steps: int) -> Optional[pd.DataFrame]:
        return self.frame(self.name(run, sp, weights, steps, "single"))

    def best6(self, run: str, sp: str, weights: str, steps: int, scale) -> Optional[pd.Series]:
        if scale is None:
            return None
        ss = [self.score_of(self.name(run, sp, weights, steps, n)) for n in self.sample_names(scale)]
        if any(s is None for s in ss):
            return None
        return pd.concat(ss, axis=1).max(axis=1)

    def diversity(self, run: str, sp: str, weights: str, steps: int, scale, tokens) -> Optional[dict]:
        folders = [self.folder(run, sp, weights, steps, n) for n in self.sample_names(scale)]
        if not all(os.path.exists(os.path.join(f, "shard_000_of_001.jsonl")) for f in folders):
            return None
        return diversity([read_paths(f) for f in folders], list(tokens))

    def config(self, run: str, sp: str, weights: str, steps: int, scale, base: Optional[dict] = None) -> dict:
        """Single path and best-of-6 of one configuration; ``base`` = the same for another configuration to pair
        against (the series are returned under ``_single`` / ``_best6`` for that)."""
        rec = {}
        d = self.single(run, sp, weights, steps)
        if d is not None:
            s = d["score"].astype(float)
            rec["single"] = {**self.stats(s, sp), "subscores": self.subscores(d)}
            rec["_single"] = s
        b = self.best6(run, sp, weights, steps, scale)
        if b is not None:
            rec["best_of_6"] = self.stats(b, sp)
            div = self.diversity(run, sp, weights, steps, scale, b.index)
            if div is not None:
                rec["best_of_6"]["diversity"] = div
            rec["_best6"] = b
        if base:
            for k, key in (("single", "_single"), ("best_of_6", "_best6")):
                if key in rec and key in base:
                    rec[k]["vs_start"] = self.paired(rec[key], base[key], sp)
        return rec

    # ---- sections -------------------------------------------------------------------------------------------
    def chosen(self, run: str) -> Optional[dict]:
        f = os.path.join(self.heads, run, "choices_navdev.json")
        if not os.path.exists(f):
            return None
        c = json.load(open(f))
        return c if "epoch" in c else None  # RL runs' choice files carry "checkpoint" and go in rl()

    def at_choice(self, run: str) -> Dict[str, dict]:
        if run not in self._at:
            c = self.chosen(run)
            self._at[run] = {} if c is None else {
                sp: self.config(run, sp, f"epoch_{c['epoch']}", c["steps"], c.get("scale")) for sp in SPLITS}
        return self._at[run]

    def runs(self) -> dict:
        out = {}
        for cf in sorted(glob.glob(os.path.join(self.heads, "*", "choices_navdev.json"))):
            run = os.path.basename(os.path.dirname(cf))
            c = self.chosen(run)
            if c is None:
                continue
            rec = {"choice": {k: c.get(k) for k in ("epoch", "steps", "scale", "navdev_single_pdms",
                                                     "navdev_best_of_6")}}
            for sp, r in self.at_choice(run).items():
                for k in ("single", "best_of_6"):
                    if k in r:
                        rec[f"{sp}_{k}"] = r[k]
            logf = os.path.join(self.heads, run, "log.jsonl")
            if os.path.exists(logf):
                for line in open(logf):
                    r = json.loads(line)
                    if r.get("epoch") == c["epoch"] and "dev_ade" in r:
                        rec["dev_ade"], rec["dev_best6_ade"] = r["dev_ade"], r.get("dev_best6_ade")
            rj = os.path.join(self.heads, run, "run.json")
            if os.path.exists(rj):
                j = json.load(open(rj))
                cfg = j.get("config") or {}
                rec.update(kind=cfg.get("kind"), layers=cfg.get("layers"), params=j.get("params"),
                           train_seconds=j.get("elapsed_s"))
            out[run] = rec
        return out

    def method_files(self) -> Dict[str, dict]:
        return {os.path.basename(f)[:-5]: json.load(open(f))
                for f in sorted(glob.glob(os.path.join(self.heads, "choices", "*.json")))}

    def pair(self, run1: str, run2: str) -> dict:
        """run1 minus run2, each at its own navdev choices, paired on the same scenes."""
        a, b = self.at_choice(run1), self.at_choice(run2)
        out = {}
        for sp in SPLITS:
            for k, key in (("single", "_single"), ("best_of_6", "_best6")):
                if key in a.get(sp, {}) and key in b.get(sp, {}):
                    out[f"{sp}_{k}"] = self.paired(a[sp][key], b[sp][key], sp)
        return out

    def layer_tests(self, methods: Dict[str, dict]) -> dict:
        out = {}
        for m, f in methods.items():
            cand = f.get("candidates") or {}
            l24 = [r for r in cand if r.endswith("-l24")]
            l3 = [r for r in cand if r.endswith("-l12-18-24")]
            if l24 and l3:
                out[m] = {"runs": [l3[0], l24[0]], "chosen": f.get("run"), **self.pair(l3[0], l24[0])}
        return out

    def series(self, methods: Dict[str, dict]) -> dict:
        """Phase 2 (R17): every run of A's continuity series (A0, A2-A4), its leave-one-out runs and its bin-size runs,
        each minus A1, each at its own navdev choices, paired on the same scenes. ``measured_effect`` is R17's call on
        navtest's single path: the difference exceeds 1.4 points and its drive interval excludes 0. Every other
        measure carries the same test as ``beyond_seed_noise``, for reporting."""
        a = methods.get("A")
        if not isinstance(a, dict) or not str(a.get("run", "")).startswith("a1-"):
            return {}
        a1 = a["run"]
        tag = a1[len("a1-"):]
        pat = re.compile(rf"^(a[0-4])(?:-(no-[a-z0-9]+|bin\d+))?-{re.escape(tag)}$")
        out = {}
        for cf in sorted(glob.glob(os.path.join(self.heads, "*", "choices_navdev.json"))):
            run = os.path.basename(os.path.dirname(cf))
            m = pat.match(run)
            if not m or run == a1 or self.chosen(run) is None:
                continue
            rec = {"config": m.group(1), "variant": m.group(2) or "", "against": a1, **self.pair(run, a1)}
            for k, v in rec.items():
                if k.startswith("nav") and isinstance(v, dict):
                    d, (lo, hi) = v["difference"], v["ci_drives"]
                    v["beyond_seed_noise"] = bool(abs(d) > R17_THRESHOLD and (lo > 0 or hi < 0))
            if "navtest_single" in rec:
                rec["measured_effect"] = rec["navtest_single"]["beyond_seed_noise"]
            out[run] = rec
        return out

    def selectors(self) -> dict:
        """Phase 6 (R14): every trained selector (``selector-<chain>-<variant>``, the chain being a contender such as
        a4 or the rehearsal): its held-out choice (epoch and own-pick margin), its training, and R14's measures on
        navdev, and on navtest once run. The adoption and final-system files are in ``methods``."""
        out = {}
        for cf in sorted(glob.glob(os.path.join(self.heads, "selector-*", "choice.json"))):
            d = os.path.dirname(cf)
            chain, _, variant = os.path.basename(d)[len("selector-"):].partition("-")
            if not variant.startswith(("xfit", "infold")):
                continue
            ch = json.load(open(cf))
            rec = {"chain": chain, "variant": variant,
                   "heldout": {k: ch.get(k) for k in ("epoch", "delta", "heldout_selected_pdms", "epochs_done")}}
            rj = os.path.join(d, "run.json")
            if os.path.exists(rj):
                r = json.load(open(rj))
                rec["train"] = {"params": r.get("params"), "scenes": r.get("scenes"), "elapsed_s": r.get("elapsed_s"),
                                "peak_gpu_gb": r.get("peak_gpu_gb"), "candidates_per_scene": r.get("candidates_per_scene"),
                                "extra_candidates_per_scene": r.get("extra_candidates_per_scene")}
            for sp in SPLITS:
                f = os.path.join(d, sp, "report.json")
                if os.path.exists(f):
                    s = json.load(open(f))
                    rep = s.get("report", {})
                    rec[sp] = {"scenes_selected": s.get("scenes"), "weights": s.get("weights"), "delta": s.get("delta"),
                               **{k: rep[k] for k in SELECTOR_KEYS if k in rep}}
            out[os.path.basename(d)] = rec
        return out

    def selector_candidates(self) -> dict:
        """Phase 6 (R14): the headroom of each chain's completely scored candidate sets (``selector-<chain>/
        candidates_<set>.jsonl`` with its ``.done`` marker): the own pick, the best of the candidates, the mean
        candidate, how often some candidate beats the own pick, and the own pick's NC or DAC failures some candidate
        passes. These are training scenes (out of fold, or in fold for the control)."""
        out = {}
        for f in sorted(glob.glob(os.path.join(self.heads, "selector-*", "candidates_*.jsonl"))):
            if not os.path.exists(f + ".done"):
                continue
            chain = os.path.basename(os.path.dirname(f))[len("selector-"):]
            name = os.path.basename(f)[len("candidates_"):-len(".jsonl")]
            out.setdefault(chain, {})[name] = headroom(f)
        return out

    def selector_halves(self) -> dict:
        """R14's measure "the half generators against the deployed one, on navdev" (reporting): per chain spec, each
        half generator's navdev single path and the deployed generator's, and deployed minus the two halves' mean,
        paired per scene, with its drive interval. Only scores already made are read (the driver makes them)."""
        out = {}
        for f in sorted(glob.glob(os.path.join(self.heads, "choices", "selector_spec_*.json"))):
            x = os.path.basename(f)[len("selector_spec_"):-len(".json")].lower()
            spec = json.load(open(f))
            dep = self.score_of(self.name(spec["deployed_run"], "navdev", spec["deployed_weights"], spec["steps"], "single"))
            halves = {}
            for h in (0, 1):
                run, w = ((f"selector-{x}-gen-half{h}-rl", f"rl_{spec['updates']}") if spec["updates"] > 0
                          else (f"selector-{x}-gen-half{h}", f"epoch_{spec['rl_start_epoch']}"))
                s = self.score_of(self.name(run, "navdev", w, spec["steps"], "single"))
                if s is not None:
                    halves[f"half{h}"] = s
            if dep is None or not halves:
                continue
            rec = {"deployed": round(float(dep.mean()), 4), **{k: round(float(v.mean()), 4) for k, v in halves.items()}}
            if len(halves) == 2:
                rec["deployed_minus_halves"] = self.paired(dep, (halves["half0"] + halves["half1"]) / 2, "navdev")
            out[x] = rec
        return out

    def selector_oof_pairs(self, train_meta: Optional[str]) -> dict:
        """Every pair of chains whose two out-of-fold candidate sets are completely scored (``oof_pair.py``; reporting
        only): A minus B for the own pick, the best candidate and the mean candidate, paired on the same training
        scenes (no half generator of either chain trained on them), pooled over both halves, with drive intervals."""
        if not train_meta or not os.path.exists(train_meta):
            return {}
        done = lambda c: all(os.path.exists(os.path.join(self.heads, f"selector-{c}", f"candidates_oof_half{h}.jsonl.done"))  # noqa: E731
                             for h in (0, 1))
        chains = sorted(c for c in (os.path.basename(d)[len("selector-"):]
                                    for d in glob.glob(os.path.join(self.heads, "selector-*"))) if done(c))
        if len(chains) < 2:
            return {}
        import oof_pair
        drives = oof_pair.token_drives(train_meta)
        per = {}
        for c in chains:
            per[c] = {}
            for h in (0, 1):
                per[c].update(oof_pair.per_scene(os.path.join(self.heads, f"selector-{c}", f"candidates_oof_half{h}.jsonl")))
        return {f"{a}-{b}": oof_pair.paired(per[a], per[b], drives) for a, b in combinations(chains, 2)}

    def comparisons(self, methods: Dict[str, dict]) -> dict:
        chosen = {m: f["run"] for m, f in methods.items() if "run" in f and "choice" in f}
        return {f"{m1}-{m2}": {"runs": [chosen[m1], chosen[m2]], **self.pair(chosen[m1], chosen[m2])}
                for m1, m2 in combinations(sorted(chosen), 2)}

    def weights_dir(self, run: str, weights: str) -> str:
        """The weights label whose navdev folder exists (an RL start of ``final`` is the last epoch's folder)."""
        if weights == "final" and not glob.glob(os.path.join(self.heads, run, "navdev_final")):
            eps = sorted(int(os.path.basename(p)[len("navdev_epoch_"):])
                         for p in glob.glob(os.path.join(self.heads, run, "navdev_epoch_*")))
            return f"epoch_{eps[-1]}" if eps else weights
        return weights

    def rl(self) -> dict:
        out = {}
        for rj in sorted(glob.glob(os.path.join(self.heads, "*", "run.json"))):
            rd = os.path.dirname(rj)
            run = os.path.basename(rd)
            j = json.load(open(rj))
            if "start" not in j or "args" not in j:
                continue
            a = j["args"]
            start_run = os.path.basename(a["run"].rstrip("/"))
            start_w = self.weights_dir(start_run, a["weights"])
            steps, scale = a["steps"], a.get("scale")
            ups = sorted(int(os.path.basename(f)[3:-3]) for f in glob.glob(os.path.join(rd, "rl_*.pt")))
            rec = {"start": {"run": start_run, "weights": start_w},
                   "settings": {**{k: a.get(k) for k in ("lr", "kl_beta", "g", "scenes_per_update", "updates", "steps",
                                                         "scale", "seed")},
                                "adv": a.get("adv", "grpo"), "bc_balance": a.get("bc_balance", 0.0),
                                "lp_sd_floor": a.get("lp_sd_floor", 0.0)},  # R13's options; R12's runs predate them
                   "rl_scenes": j.get("rl_scenes"), "checkpoints": {}}
            if j.get("bc"):
                rec["bc_weight"] = j["bc"].get("weight")
            base = self.config(start_run, "navdev", start_w, steps, scale)
            for u in [0] + ups:
                r = base if u == 0 else self.config(run, "navdev", f"rl_{u}", steps, scale, base=base)
                rec["checkpoints"][str(u)] = {k: v for k, v in r.items() if not k.startswith("_")}
            logf = os.path.join(rd, "log.jsonl")
            if os.path.exists(logf):
                L = [json.loads(line) for line in open(logf) if line.strip()]
                rec["curve"] = [{"updates": [w[0]["update"], w[-1]["update"]],
                                 **{k: round(float(np.mean([x[k] for x in w])), 4) for k in CURVE if k in w[0]}}
                                for w in (L[i:i + 50] for i in range(0, len(L), 50))]
                rec["updates_done"] = L[-1]["update"] if L else 0
            cf = os.path.join(rd, "choices_navdev.json")
            if os.path.exists(cf):
                rec["choice"] = json.load(open(cf))
            out[run] = rec
        return out

    def collect(self, train_meta: Optional[str] = None) -> dict:
        methods = self.method_files()
        out = {"generated": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}", "references": {}}
        for sp in SPLITS:
            for label, name in (("h1_text", REF[sp]), ("constant_velocity", CV[sp])):
                d = self.frame(name)
                if d is not None:
                    s = d["score"].astype(float)
                    g = np.array([self.drives[sp][t] for t in s.index])
                    out["references"][f"{label}_{sp}"] = {"pdms": round(float(s.mean()), 4),
                                                          "ci_drives": drive_ci(s.values, g),
                                                          "scenes": int(len(s)), "subscores": self.subscores(d)}
        out.update(runs=self.runs(), layer_tests=self.layer_tests(methods), comparisons=self.comparisons(methods),
                   series=self.series(methods), rl=self.rl(), selectors=self.selectors(),
                   selector_candidates=self.selector_candidates(), selector_halves=self.selector_halves(),
                   selector_oof_pairs=self.selector_oof_pairs(train_meta), methods=methods)
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--exp", default=f"{HOME}/navsim_workspace/exp")
    p.add_argument("--cache-dir", default=f"{HOME}/navsim_workspace/head_cache",
                   help="the folder with drives_navdev.json and drives_navtest.json")
    p.add_argument("--train-meta", default=f"{HOME}/navsim_workspace/head_cache/h1_l12-18-24/train/meta.jsonl",
                   help="a stored-feature cache's training meta.jsonl, for the training scenes' drives (out-of-fold pairs)")
    args = p.parse_args()
    out = Results(args.exp, args.cache_dir).collect(args.train_meta)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{len(out['runs'])} runs, {len(out['rl'])} RL runs, {len(out['comparisons'])} method pairs, "
          f"{len(out['methods'])} method files -> {args.out}")


if __name__ == "__main__":
    main()
