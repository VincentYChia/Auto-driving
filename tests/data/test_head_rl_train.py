"""``pilot_sft/heads/rl_head.py`` end to end on CPU: a tiny trained path generator, a fake PDM server in a thread
(reward from the path), three RL updates, checkpoints in the supervised format, and ``predict_head`` loading one."""

import json
import socket
import subprocess
import sys
import threading
from multiprocessing.connection import Listener
from pathlib import Path

import pytest

pytest.importorskip("torch")
HERE = Path(__file__).resolve().parent
HEADS = HERE.parents[1] / "pilot_sft" / "heads"
sys.path.insert(0, str(HEADS))
sys.path.insert(0, str(HERE))

from test_head_train import make_split, run  # noqa: E402


def fake_server(port, stop):
    with Listener(("127.0.0.1", port), authkey=b"heads-pdm") as listener:
        listener._listener._socket.settimeout(30)
        conn = listener.accept()
        while not stop.is_set():
            try:
                msg = conn.recv()
            except EOFError:
                break
            if msg.get("op") == "close":
                break
            res = [[{"score": min(1.0, max(0.0, p[-1][0] / 20.0)), "ego_progress": 0.5} for p in paths]
                   for _, paths in msg["items"]]
            conn.send({"results": res, "seconds": 0.0})
        conn.close()


@pytest.mark.parametrize("kind,extra", [("masked", []), ("anchor", ["--anchors", "4"])])
def test_rl_updates_and_checkpoints(tmp_path, kind, extra):
    cache = tmp_path / "cache"
    make_split(cache / "train", 24, seed=0)
    make_split(cache / "dev", 8, seed=1)
    out = tmp_path / "sft"
    run("train_head.py", "--cache", str(cache), "--out", str(out), "--kind", kind, "--width", "32", "--depth", "2",
        "--heads", "4", "--batch", "8", "--warmup", "2", "--workers", "1", "--epochs", "1", *extra)
    view = tmp_path / "rlview"
    view.mkdir()
    (view / "train.jsonl").write_text("".join(json.dumps({"id": f"t{i}"}) + "\n" for i in range(0, 24, 2)))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    stop = threading.Event()
    th = threading.Thread(target=fake_server, args=(port, stop), daemon=True)
    th.start()
    rl = tmp_path / "rl"
    run("rl_head.py", "--run", str(out), "--weights", "epoch_1", "--out", str(rl), "--cache", str(cache / "train"),
        "--rl-scenes", str(view), "--steps", "2", "--scale", "1.0", "--updates", "3", "--scenes-per-update", "4",
        "--g", "3", "--save-every", "2", "--port", str(port))
    stop.set()
    log = [json.loads(l) for l in (rl / "log.jsonl").read_text().splitlines()]
    assert [r["update"] for r in log] == [1, 2, 3]
    assert all(0.0 <= r["reward"] <= 1.0 and r["scorer_errors"] == 0 for r in log)
    for f in ("rl_2.pt", "rl_3.pt", "checkpoint.pt", "run.json"):
        assert (rl / f).exists(), f
    assert json.loads((rl / "run.json").read_text())["rl_scenes"] == 12
    run("predict_head.py", "--run", str(rl), "--weights", "rl_3", "--cache", str(cache / "dev"), "--split", "navdev",
        "--steps", "2", "--k", "2")
    assert (rl / "navdev_rl_3" / "steps2" / "single" / "shard_000_of_001.jsonl").exists()
