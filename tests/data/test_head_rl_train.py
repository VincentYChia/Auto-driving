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


@pytest.mark.parametrize("kind,extra,rl_extra", [
    ("masked", [], []),
    ("anchor", ["--anchors", "4"], ["--kl-beta", "0.04"]),
    ("masked", [], ["--adv", "dr", "--bc-balance", "1"]),  # R13's RL6 on A
    ("anchor", ["--anchors", "4"], ["--adv", "dr", "--bc-balance", "1", "--lp-sd-floor", "0.1"]),  # R13's RL6 on D
    ("ddim", [], ["--adv", "dr", "--lp-sd-floor", "0.1"]),  # phase 4's RL4 on B
    ("flow", [], ["--adv", "dr", "--lp-sd-floor", "0.1"]),  # phase 4's RL4 on C
    ("ddim", [], ["--adv", "dr", "--bc-balance", "1", "--lp-sd-floor", "0.1"]),  # RL6 on a continuous method
])
def test_rl_updates_and_checkpoints(tmp_path, kind, extra, rl_extra):
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
        "--g", "3", "--save-every", "2", "--port", str(port), *rl_extra)
    stop.set()
    log = [json.loads(l) for l in (rl / "log.jsonl").read_text().splitlines()]
    assert [r["update"] for r in log] == [1, 2, 3]
    assert all(0.0 <= r["reward"] <= 1.0 and r["scorer_errors"] == 0 for r in log)
    if "--kl-beta" in rl_extra:  # with the KL term the policy starts at the reference, then drifts a little
        assert log[0]["kl"] < 1e-4 and all(r["kl"] >= 0 for r in log)
    meta = json.loads((rl / "run.json").read_text())
    if "--bc-balance" in rl_extra:  # R13: the BC weight was measured at the start, and every update has the BC term
        assert meta["bc"]["weight"] > 0 and len(meta["bc"]["rl_grad_norms"]) == 4 and all(r["bc"] > 0 for r in log)
        assert meta["args"]["adv"] == "dr"
    else:
        assert "bc" not in meta and all("bc" not in r for r in log)
    for f in ("rl_2.pt", "rl_3.pt", "checkpoint.pt", "run.json"):
        assert (rl / f).exists(), f
    assert json.loads((rl / "run.json").read_text())["rl_scenes"] == 12
    run("predict_head.py", "--run", str(rl), "--weights", "rl_3", "--cache", str(cache / "dev"), "--split", "navdev",
        "--steps", "2", "--k", "2")
    assert (rl / "navdev_rl_3" / "steps2" / "single" / "shard_000_of_001.jsonl").exists()


def test_advantages_with_and_without_the_group_sd():
    """R13: the SD division turns a group's tiny progress gaps into full-size advantages; Dr. GRPO keeps them in reward
    units. A group whose rewards are all equal gives no gradient either way."""
    import torch

    import rl_head
    r = torch.tensor([[0.9, 0.9, 0.9, 0.9], [0.80, 0.81, 0.82, 0.83], [0.0, 1.0, 1.0, 1.0]])
    a, d = rl_head.advantages(r, "grpo"), rl_head.advantages(r, "dr")
    assert torch.all(a[0] == 0) and torch.all(d[0] == 0)
    assert a[1].abs().max() > 1.0 and d[1].abs().max() < 0.02
    assert torch.allclose(d[2], torch.tensor([-0.75, 0.25, 0.25, 0.25]))
    assert torch.allclose(a.sum(1), torch.zeros(3), atol=1e-6) and torch.allclose(d.sum(1), torch.zeros(3), atol=1e-6)
