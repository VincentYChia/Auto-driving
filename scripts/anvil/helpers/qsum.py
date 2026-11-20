# Backup copy of qsum.py from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Queue summary from sacct -P (JobID|Partition|Eligible|Start|Elapsed|State|AllocTRES), allocations only.
import sys, datetime as dt, statistics as st
def t(s):
    try: return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception: return None
def el(s):
    d = 0
    if "-" in s: d, s = s.split("-"); d = int(d)
    h, m, x = (s.split(":") + ["0"] * 3)[:3] if s.count(":") == 2 else (["0"] + s.split(":") + ["0"])[:3]
    return d * 24 + int(h) + int(m) / 60 + float(x) / 3600
rows = {}
for line in open(sys.argv[1]):
    f = line.rstrip("\n").split("|")
    if len(f) < 6 or "." in f[0] or f[0] == "JobID": continue
    p, e, s, l, state = f[1], t(f[2]), t(f[3]), f[4], f[5]
    if not e or not s: continue
    rows.setdefault(p.split(",")[0], []).append(((s - e).total_seconds() / 3600, el(l)))
for p, v in sorted(rows.items()):
    w = [a for a, _ in v]; r = [b for _, b in v]
    print(f"{p:10s} started {len(v):3d}  wait h total {sum(w):7.1f}  run h total {sum(r):6.1f}  ratio {sum(w)/max(sum(r),1e-9):5.1f}:1  median wait {st.median(w):5.1f} h  max {max(w):5.1f} h")
