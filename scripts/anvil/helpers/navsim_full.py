# Backup copy of navsim_full.py from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Usage: navsim_full.py <score dir name> <prediction dir> <what> <job> <out json>
import glob, json, sys, numpy as np, pandas as pd
E="/anvil/scratch/x-vchia/navsim_workspace/exp/"
name, pred, what, job, outfn = sys.argv[1:6]
f=sorted(glob.glob(E+name+"_navtest_v1/*/*.csv"))[-1]; d=pd.read_csv(f); d=d[d.token!="average"].set_index("token")
c=pd.read_csv(E+"cv_navtest_v1/2026.11.03.18.40.00/2026.11.03.18.41.38.csv"); c=c[c.token!="average"].set_index("token")
rows=[json.loads(l) for p in sorted(glob.glob(pred+"/shard_*.jsonl")) for l in open(p) if l.strip()]
valid=d.valid.astype(bool); print(f, "rows", len(d), "valid", int(valid.sum()), "predictions", len(rows), "unique", len({r["token"] for r in rows}), "fallbacks", sum(1 for r in rows if r.get("fallback")))
assert len(d)==12146 and valid.all()
cols=["no_at_fault_collisions","drivable_area_compliance","time_to_collision_within_bound","comfort","ego_progress","driving_direction_compliance","score"]
s=d[cols].astype(float); m=s.mean().round(4).to_dict(); m["pdms"]=m.pop("score")
rng=np.random.default_rng(0); x=s.score.values
b=[rng.choice(x,len(x)).mean() for _ in range(5000)]
ids=s.index.intersection(c.index); dd=x-c.loc[s.index,"score"].astype(float).values
out={"what":what,"evidence":"V (measured on Anvil)","split":"navtest","devkit":"navsim v1.1","scenes":len(d),"failed":int((~valid).sum()),
 "predictions":len(rows),"fallbacks_to_constant_velocity":sum(1 for r in rows if r.get("fallback")),"job":job,"metrics":m,
 "pdms_95ci_bootstrap":[round(np.percentile(b,2.5),4),round(np.percentile(b,97.5),4)],
 "share_scenes_score_zero":round(float((x==0).mean()),4),"median_scene_score":round(float(np.median(x)),4),
 "share_zero_by_metric":{k:round(float((s[k]==0).mean()),4) for k in ("no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance")},
 "vs_constant_velocity_same_scenes":{"pdms_difference":round(float(dd.mean()),4)}}
json.dump(out,open(outfn,"w"),indent=1); print(json.dumps(out,indent=1))
