# Backup copy of paired.py from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Paired NAVSIM comparison on the same scenes: paired.py <score name A> <score name B> [<score name B2> ...]
# Prints A - B for PDMS and each sub-score, with a 95% bootstrap CI over scenes (5,000 resamples, seed 0),
# and the share of zero-score scenes for each.
import glob, sys
import numpy as np, pandas as pd
E = "/anvil/scratch/x-vchia/navsim_workspace/exp/"
def load(name):
    f = sorted(glob.glob(E + name + "_navtest_v1/*/*.csv"))[-1]
    d = pd.read_csv(f); d = d[d.token != "average"].set_index("token")
    assert len(d) == 12146 and d.valid.astype(bool).all(), (name, len(d))
    return d
cols = ["score", "no_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_progress", "comfort"]
a = load(sys.argv[1])
for other in sys.argv[2:]:
    b = load(other); ids = a.index.intersection(b.index)
    print(f"== {sys.argv[1]} - {other}  scenes {len(ids)}")
    for c in cols:
        d = a.loc[ids, c].astype(float).values - b.loc[ids, c].astype(float).values
        rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        print(f"  {c:32s} {d.mean():+.4f}  [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    za = float((a.loc[ids, "score"] == 0).mean()); zb = float((b.loc[ids, "score"] == 0).mean())
    print(f"  zero-score share                 {za:.4f} vs {zb:.4f}")
