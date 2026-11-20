"""Run on the desktop: python pilot_sft/heads/rl_by_command.py > rl_by_command.json. navdev PDMS, DAC and NC failures and progress by route command: H1's text output, A's and D's RL start points and
their R13 recipe runs (RL4, RL6) at 500 updates. Prints one JSON object."""
import glob
import json
import os

import numpy as np
import pandas as pd

EXP = os.path.expanduser("~/navsim_workspace/exp")
H = os.path.join(EXP, "heads")
CACHE = os.path.expanduser("~/navsim_workspace/head_cache/h1_l12-18-24/dev")
COMMANDS = ("turn left", "go straight", "turn right", "none given")


def frame(name):
    f = sorted(glob.glob(os.path.join(EXP, name, "*", "*.csv")))
    if not f:
        return None
    d = pd.read_csv(f[-1])
    return d[d.token != "average"].set_index("token")


cmd = {}
with open(os.path.join(CACHE, "meta.jsonl")) as f:
    for line in f:
        m = json.loads(line)
        cmd[m["id"].rsplit("/", 1)[-1]] = COMMANDS[int(np.argmax(m["ego"][-4:]))]
runs = {"H1 text": "h1-text_navdev_v1",
        "A start (epoch 20)": "heads-a1-l12-18-24-navdev_epoch_20-steps8-single_navdev_v1",
        "A RL4 500": "heads-a1-l12-18-24-rlrecipe4-navdev_rl_500-steps8-single_navdev_v1",
        "A RL6 500": "heads-a1-l12-18-24-rlrecipe6-navdev_rl_500-steps8-single_navdev_v1",
        "D start (epoch 5)": "heads-d-l12-18-24-navdev_epoch_5-steps2-single_navdev_v1",
        "D RL4 500": "heads-d-l12-18-24-rlrecipe4-navdev_rl_500-steps2-single_navdev_v1",
        "D RL6 250": "heads-d-l12-18-24-rlrecipe6-navdev_rl_250-steps2-single_navdev_v1"}
out = {}
for label, name in runs.items():
    d = frame(name)
    if d is None:
        out[label] = None
        continue
    d["command"] = [cmd.get(t, "unknown") for t in d.index]
    res = {"all": {"pdms": round(100 * float(d["score"].mean()), 2), "scenes": int(len(d))}}
    for k, x in d.groupby("command"):
        res[k] = {"scenes": int(len(x)), "pdms": round(100 * float(x["score"].mean()), 2),
                  "dac_fail_pct": round(100 * float((x["drivable_area_compliance"] == 0).mean()), 2),
                  "nc_fail_pct": round(100 * float((x["no_at_fault_collisions"] == 0).mean()), 2),
                  "ep": round(100 * float(x["ego_progress"].mean()), 2),
                  "ttc": round(100 * float(x["time_to_collision_within_bound"].mean()), 2)}
    out[label] = res
s = json.load(open(os.path.join(H, "choices", "A_seeds.json"))) if os.path.exists(os.path.join(H, "choices", "A_seeds.json")) else None
out["A_seeds"] = s
print(json.dumps(out))
