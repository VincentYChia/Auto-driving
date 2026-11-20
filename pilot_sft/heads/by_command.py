"""Run on the desktop: python pilot_sft/heads/by_command.py > by_command.json. navtest PDMS and sub-scores by route command for H1's text output and each path generator's own pick (layers
12/18/24, at each run's choices; R at both layer settings). Prints one JSON object."""
import glob
import json
import os

import numpy as np
import pandas as pd

EXP = os.path.expanduser("~/navsim_workspace/exp")
H = os.path.join(EXP, "heads")
CACHE = os.path.expanduser("~/navsim_workspace/head_cache/h1_l12-18-24/navtest")
COMMANDS = ("turn left", "go straight", "turn right", "none given")


def frame(name):
    f = sorted(glob.glob(os.path.join(EXP, name, "*", "*.csv")))
    d = pd.read_csv(f[-1])
    return d[d.token != "average"].set_index("token")


cmd = {}
with open(os.path.join(CACHE, "meta.jsonl")) as f:
    for line in f:
        m = json.loads(line)
        cmd[m["id"].rsplit("/", 1)[-1]] = COMMANDS[int(np.argmax(m["ego"][-4:]))]
runs = {"H1 text": "deskcheck-h1_navtest_v1"}
for run in ("a1-l12-18-24", "b-l12-18-24", "c-l12-18-24", "d-l12-18-24", "r-l12-18-24", "r-l24"):
    c = json.load(open(os.path.join(H, run, "choices_navdev.json")))
    runs[run] = f"heads-{run}-navtest_epoch_{c['epoch']}-steps{c['steps']}-single_navtest_v1"
out = {"scenes_by_command": {}}
for label, name in runs.items():
    d = frame(name)
    d["command"] = [cmd.get(t, "unknown") for t in d.index]
    out["scenes_by_command"] = {k: int(v) for k, v in d["command"].value_counts().items()}
    g = d.groupby("command")
    out[label] = {k: {"pdms": round(100 * float(x["score"].mean()), 2),
                      "dac_fail_pct": round(100 * float((x["drivable_area_compliance"] == 0).mean()), 2),
                      "nc_fail_pct": round(100 * float((x["no_at_fault_collisions"] == 0).mean()), 2),
                      "ep": round(100 * float(x["ego_progress"].mean()), 2)} for k, x in g}
print(json.dumps(out))
