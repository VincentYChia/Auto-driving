"""Score every candidate path of one prediction folder with the PDM scoring service: phase 6's selector data.

    python pilot_sft/heads/pdm_server.py serve --cache <navtrain metric cache> --workers 12 --port 47411 &
    python pilot_sft/heads/score_candidates.py --paths <run>/navtrain_<weights>/steps<S> --out <file>.jsonl

``--paths`` has ``predict_head.py``'s layout: ``single/`` and the ``sample_<j>/`` or ``scale<x>_sample_<j>/``
folders. The output has one line per scene, ``{"token", "names", "results"}``: the candidate folders' names, single
first, and the server's result for each (``score``, the PDMS in [0, 1], and the sub-scores; ``score`` 0 and
``error`` when NAVSIM could not score a path). The server scores each path exactly as NAVSIM's ``run_pdm_score.py``
does (``pdm_server.py check``). Scenes already in ``--out`` are skipped, so a stopped run resumes; a line torn by
the stop is dropped first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdm_client import PDMClient  # noqa: E402


def candidate_names(paths: str) -> List[str]:
    names = sorted(d for d in os.listdir(paths) if os.path.isfile(os.path.join(paths, d, "shard_000_of_001.jsonl")))
    if not names:
        raise SystemExit(f"no candidate folders in {paths}")
    return (["single"] if "single" in names else []) + [n for n in names if n != "single"]


def read_candidates(paths: str, names: List[str]) -> Dict[str, Dict[str, list]]:
    """token -> {candidate name: waypoints}, in the order the scenes appear in the first folder."""
    out: Dict[str, Dict[str, list]] = {}
    for n in names:
        with open(os.path.join(paths, n, "shard_000_of_001.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out.setdefault(r["token"], {})[n] = r["waypoints"]
    return out


def scored_tokens(out: str) -> set:
    """Tokens already in ``out``; a torn last line (from a stop) is removed so appending stays valid JSON lines."""
    if not os.path.exists(out):
        return set()
    good, done = [], set()
    with open(out, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["token"])
                good.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError):
                pass
    with open(out + ".part", "w", encoding="utf-8") as f:
        f.writelines(good)
    os.replace(out + ".part", out)
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--paths", required=True, help="a predict_head.py steps<S> folder")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=47411)
    p.add_argument("--scenes-per-request", type=int, default=64)
    args = p.parse_args()
    names = candidate_names(args.paths)
    cands = read_candidates(args.paths, names)
    incomplete = [t for t, c in cands.items() if len(c) != len(names)]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} scenes lack a path in some candidate folder, e.g. {incomplete[0]}")
    done = scored_tokens(args.out)
    todo = [t for t in cands if t not in done]
    print(f"{len(cands)} scenes x {len(names)} candidates ({', '.join(names)}); {len(done)} already scored, "
          f"{len(todo)} to go", flush=True)
    client, t0, n_paths, errors = PDMClient(args.port), time.time(), 0, 0
    try:
        with open(args.out, "a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.scenes_per_request):
                chunk = todo[i : i + args.scenes_per_request]
                res = client.score([(t, [cands[t][n] for n in names]) for t in chunk])
                for t, r in zip(chunk, res):
                    f.write(json.dumps({"token": t, "names": names, "results": r}) + "\n")
                    errors += sum(1 for x in r if "error" in x)
                f.flush()
                n_paths += len(chunk) * len(names)
                if (i // args.scenes_per_request) % 50 == 0:
                    el = time.time() - t0  # can be 0 on a coarse clock (Windows: ~16 ms) after a fast first request
                    print(json.dumps({"scenes_done": len(done) + i + len(chunk),
                                      "paths_per_s": round(n_paths / el, 1) if el > 0 else None, "errors": errors}),
                          flush=True)
    finally:
        client.close()
    el = time.time() - t0
    print(json.dumps({"scenes": len(cands), "scored_now": len(todo), "paths": n_paths, "seconds": round(el, 1),
                      "paths_per_s": round(n_paths / el, 1) if el > 0 else None, "errors": errors}), flush=True)


if __name__ == "__main__":
    main()
