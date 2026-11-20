"""End to end on CPU: a tiny synthetic cache, ``train_head.py`` for 2 epochs (every epoch saved), a resume, and
``predict_head.py`` writing NAVSIM-format paths for each step count and sample."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
HEADS = Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"
sys.path.insert(0, str(HEADS))

import features  # noqa: E402


def make_split(folder, rows, targets=True, seed=0, prefix=""):
    rng = np.random.default_rng(seed)
    w = features.Writer(str(folder), [12, 24], rows=rows, tokens=6, width=16, settings={"model": "synthetic"})
    for s in range(0, rows, 4):
        b = min(4, rows - s)
        feats = {l: rng.integers(0, 2**15, size=(b, 6, 16), dtype=np.uint16) & 0x3FFF for l in (12, 24)}
        metas = []
        for i in range(b):
            m = {"id": f"{prefix}t{s + i}", "ego": rng.normal(size=features.EGO_DIM).tolist()}
            if targets:
                m["target_xy"] = np.cumsum(rng.uniform(0, 2, size=(8, 2)) * [1, 0.1], axis=0).tolist()
                m["target_yaw"] = [0.0] * 8
            metas.append(m)
        w.write(feats, [6] * b, metas)
    w.finish({})


def run(script, *args):
    r = subprocess.run([sys.executable, str(HEADS / script), *args], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


@pytest.mark.parametrize("kind,extra", [("masked", ["--soft-sigma", "1", "--l1-weight", "1", "--offset"]),
                                        ("anchor", ["--anchors", "4"]), ("flow", []), ("regression", [])])
def test_train_resume_predict(tmp_path, kind, extra):
    cache = tmp_path / "cache"
    make_split(cache / "train", 24, seed=0)
    make_split(cache / "dev", 8, seed=1)
    make_split(cache / "navtest", 6, targets=False, seed=2)
    out = tmp_path / "run"
    common = ["--cache", str(cache), "--out", str(out), "--kind", kind, "--width", "32", "--depth", "2",
              "--heads", "4", "--batch", "8", "--warmup", "2", "--workers", "1", *extra]
    run("train_head.py", *common, "--epochs", "1")
    run("train_head.py", *common, "--epochs", "2", "--resume")
    for f in ("epoch_1.pt", "epoch_2.pt", "final.pt", "checkpoint.pt", "run.json", "log.jsonl"):
        assert (out / f).exists(), f
    log = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in log if "dev_ade" in r] == [1, 2]
    run("predict_head.py", "--run", str(out), "--cache", str(cache / "navtest"), "--steps", "1,2", "--k", "3")
    for steps in (1, 2):
        for name in ("single", "sample_0", "sample_2"):
            rows = [json.loads(l) for l in (out / "navtest_final" / f"steps{steps}" / name / "shard_000_of_001.jsonl")
                    .read_text().splitlines()]
            assert [r["token"] for r in rows] == [f"t{i}" for i in range(6)]
            assert np.asarray(rows[0]["waypoints"]).shape == (8, 2)
    assert "steps2" in json.loads((out / "navtest_final" / "timing.json").read_text())
    # navdev: dev-view ids ("openscene/<log>/<token>") come out as bare tokens; a scale grid adds folders
    make_split(cache / "devids", 5, seed=3, prefix="openscene/2021.06.14.18.33.41_veh-35_00001_00002/")
    run("predict_head.py", "--run", str(out), "--split", "navdev", "--cache", str(cache / "devids"), "--weights",
        "epoch_1", "--steps", "2", "--k", "2", "--scales", "0.5,1")
    for name in ("single", "sample_1", "scale0.5_sample_0"):
        rows = [json.loads(l) for l in (out / "navdev_epoch_1" / "steps2" / name / "shard_000_of_001.jsonl")
                .read_text().splitlines()]
        assert [r["token"] for r in rows] == [f"t{i}" for i in range(5)]
