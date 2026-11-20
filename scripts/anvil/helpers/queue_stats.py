# Backup copy of queue_stats.py from ~/xv/ on Anvil (2026-11-19); the desktop address comes from ~/.ssh/desk4090.host.
# Queue numbers for progress reports: wait (eligible -> start) and run time per partition, since 2026-11-03
import subprocess, datetime as dt, statistics as st, collections
F = "%Y-%m-%dT%H:%M:%S"
out = subprocess.run(["sacct", "-X", "-P", "-n", "-S", "2026-11-03", "-u", "x-vchia", "-o",
                      "JobID,JobName,Partition,Account,Eligible,Start,End,ElapsedRaw,State"], stdout=subprocess.PIPE, universal_newlines=True).stdout
open("/home/x-vchia/xv/sacct_all.txt", "w").write(out)
now = dt.datetime.now()
started, pending = collections.defaultdict(list), collections.defaultdict(list)
for line in out.splitlines():
    jid, name, part, acct, elig, start, end, el, state = line.split("|")
    part = part.split(",")[0]
    if elig in ("Unknown", "None", ""):
        continue
    e = dt.datetime.strptime(elig, F)
    if start not in ("Unknown", "None", ""):
        s = dt.datetime.strptime(start, F)
        started[part].append(((s - e).total_seconds() / 3600, int(el or 0) / 3600))
    elif state.startswith("PENDING"):
        pending[part].append((now - e).total_seconds() / 3600)
for part in sorted(set(started) | set(pending)):
    w = [x[0] for x in started[part]]; r = [x[1] for x in started[part]]
    line = f"{part:10s} started {len(w):3d}"
    if w:
        line += f" | wait median {st.median(w):6.1f} h, min {min(w):5.1f}, max {max(w):6.1f}, total {sum(w):7.1f} h | run total {sum(r):6.1f} h"
    p = pending[part]
    if p:
        line += f" | pending {len(p)} (median so far {st.median(p):.1f} h, max {max(p):.1f})"
    print(line)
