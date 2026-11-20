# Backup copy of turns.py from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# PDMS and zero share on navtest by what the human did next (turn left / right: heading change > 0.35 rad at 4 s).
import glob, sys, json
import numpy as np, pandas as pd
E = "/anvil/scratch/x-vchia/navsim_workspace/exp/"
z = np.load(E + "failure_analysis/navtest_scenes.npz", allow_pickle=True)
dh = z["human"][:, -1, 2]
grp = pd.Series(np.where(dh > 0.35, "turn left", np.where(dh < -0.35, "turn right", "other")), index=z["tokens"])
out = {}
for name in sys.argv[1:]:
    f = sorted(glob.glob(E + name + "_navtest_v1/*/*.csv"))[-1]
    d = pd.read_csv(f); d = d[d.token != "average"].set_index("token")
    assert len(d) == 12146
    g = grp.reindex(d.index)
    row = {"all": [round(d.score.mean(), 4), round((d.score == 0).mean(), 4)]}
    for k in ["turn left", "turn right", "other"]:
        s = d.score[g == k]; row[k] = [round(s.mean(), 4), round((s == 0).mean(), 4), int(len(s))]
    out[name] = row
    print(name, row)
json.dump(out, open("/tmp/turns.json", "w"))
