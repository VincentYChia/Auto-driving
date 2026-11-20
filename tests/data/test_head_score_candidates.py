"""``pilot_sft/heads/score_candidates.py`` against a fake PDM server: every candidate of a prediction folder is scored,
single first; a second run resumes without rescoring; a line torn by a stop is dropped and rescored."""

import json
import socket
import subprocess
import sys
import threading
from multiprocessing.connection import Listener
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1] / "pilot_sft" / "heads" / "score_candidates.py"


def fake_server(port, seen):
    """Accepts connections until told to stop; the score of a path is its last x / 20, clipped to [0, 1]."""
    with Listener(("127.0.0.1", port), authkey=b"heads-pdm") as listener:
        listener._listener._socket.settimeout(60)
        while True:
            conn = listener.accept()
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                if msg.get("op") == "close":
                    break
                if msg.get("op") == "stop":
                    conn.close()
                    return
                seen.extend(t for t, _ in msg["items"])
                conn.send({"results": [[{"score": min(1.0, max(0.0, p[-1][0] / 20.0)), "no_at_fault_collisions": 1.0}
                                        for p in paths] for _, paths in msg["items"]], "seconds": 0.0})
            conn.close()


def write_folder(root, name, paths):
    d = root / name
    d.mkdir(parents=True)
    (d / "shard_000_of_001.jsonl").write_text("".join(json.dumps({"token": t, "waypoints": w}) + "\n"
                                                      for t, w in paths.items()))


def path(x):
    return [[x * (i + 1) / 8, 0.0] for i in range(8)]


def run(*args):
    r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_scores_every_candidate_and_resumes(tmp_path):
    root = tmp_path / "steps2"
    toks = [f"tok{i}" for i in range(5)]
    write_folder(root, "single", {t: path(10.0) for t in toks})
    write_folder(root, "scale1.5_sample_0", {t: path(4.0 * i) for i, t in enumerate(toks)})
    write_folder(root, "scale1.5_sample_1", {t: path(30.0) for t in toks})
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    seen = []
    th = threading.Thread(target=fake_server, args=(port, seen), daemon=True)
    th.start()
    out = tmp_path / "scores.jsonl"
    run("--paths", str(root), "--out", str(out), "--port", str(port), "--scenes-per-request", "2")
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["token"] for r in rows] == toks
    assert all(r["names"] == ["single", "scale1.5_sample_0", "scale1.5_sample_1"] for r in rows)
    assert [r["results"][0]["score"] for r in rows] == [0.5] * 5
    assert [r["results"][1]["score"] for r in rows] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert all(r["results"][2]["score"] == 1.0 for r in rows)
    assert len(seen) == 5

    log = run("--paths", str(root), "--out", str(out), "--port", str(port))  # resume: nothing left to score
    assert "5 already scored, 0 to go" in log and len(seen) == 5

    text = out.read_text().splitlines()  # a stop tore the last line: it is dropped and that scene rescored
    out.write_text("\n".join(text[:-1]) + "\n" + text[-1][:20])
    run("--paths", str(root), "--out", str(out), "--port", str(port))
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(r["token"] for r in rows) == toks and seen[-1] == "tok4" and len(seen) == 6

    from multiprocessing.connection import Client
    c = Client(("127.0.0.1", port), authkey=b"heads-pdm")
    c.send({"op": "stop"})
    c.close()
