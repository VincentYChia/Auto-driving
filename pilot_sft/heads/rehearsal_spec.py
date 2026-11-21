"""A rehearsal of phase 6's selector chain (docs/HEADS_PLAN.md R14) at full scale, before any contender reaches it.

Run on the desktop:  python pilot_sft/heads/rehearsal_spec.py [--method A]

It writes ``choices/selector_spec_rehearsal.json`` for the method's phase-1 configuration (seed 0) without RL,
deployed at its R11 starting epoch, as phase 6 writes a contender's spec when the full RL stage keeps update 0. Two
keys differ from a contender's spec:
  - ``variants``: only the cross-fitted seed-0 selector and its boundary ablation, so the in-fold set is not made;
  - ``first``: its jobs go ahead of the rest of each queue (``driver.queue_job``).
The driver then runs the chain as for any spec: both half generators, out-of-fold candidates for every RL scene, their
PDM scores, the shared boundary set (kept for every later chain), selector training, and navdev evaluation.

It is no contender: R16's final system reads only the contenders' specs. Its numbers are used for no choice, and no
selector setting is changed because of them (R14: no selector setting uses navdev). A failure it finds is fixed in the
code for every chain, before any contender's chain runs.
"""
import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import driver  # noqa: E402

VARIANTS = ["xfit-s0", "xfit-boundary"]
ROLE = ("rehearsal of R14's chain at full scale on the phase-1 configuration without RL; no contender (R16), "
        "used for no choice")


def rehearsal_spec(method: str = "A") -> dict:
    """The rehearsal's spec, written once. Needs the method's layer choice (R10) and RL starting epoch (R11)."""
    out = os.path.join(driver.P["heads"], "choices", "selector_spec_rehearsal.json")
    if os.path.exists(out):
        return driver.read_json(out)
    lc = driver.choose_layers(method)
    start = driver.rl_start(method, lc) if lc else None
    if not start:
        raise SystemExit(f"{method} has no layer choice or RL starting epoch yet")
    c = lc["choice"]
    spec = {"method": method, "kind": lc["kind"], "layers": lc["layers"], "flags": [], "rl_start_epoch": start["epoch"],
            "recipe": None, "updates": 0, "save_every": driver.FULL_RL_EVERY, "deployed_run": lc["run"],
            "deployed_weights": f"epoch_{start['epoch']}", "steps": c["steps"], "scale": c["scale"],
            "k": driver.GRID[lc["kind"]]["k"], "variants": VARIANTS, "first": True, "role": ROLE,
            "rule": "docs/HEADS_PLAN.md R14", "written": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"}
    driver.write_json(out, spec)
    driver.log(f"phase 6 rehearsal: selector spec for {lc['run']} at epoch {start['epoch']}, variants {VARIANTS}")
    return spec


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--method", default="A")
    print(json.dumps(rehearsal_spec(p.parse_args().method)))
