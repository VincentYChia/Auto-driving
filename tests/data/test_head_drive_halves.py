"""Phase 6 cross-fitting (docs/HEADS_PLAN.md): the training drives split into two halves, and a path generator trained
on one half only, with its target statistics from that half."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HEADS = HERE.parents[1] / "pilot_sft" / "heads"
sys.path.insert(0, str(HEADS))
sys.path.insert(0, str(HERE))

import drive_map  # noqa: E402

LOGS = ["2021.05.12.19.36.12_veh-35_00005_00204", "2021.06.07.12.54.00_veh-38_01100_01500",
        "2021.07.24.20.37.45_veh-17_00015_00375", "2021.08.17.18.54.02_veh-45_00665_01065",
        "2021.09.13.13.07.13_veh-26_02010_02450", "2021.10.06.17.43.07_veh-28_00508_00877"]


def ids_for(counts):
    return [f"openscene/{LOGS[d]}/tok{d}x{i}" for d, n in enumerate(counts) for i in range(n)]


def test_drive_halves_split_by_drive_balanced_by_scenes_and_fixed_by_the_seed():
    ids = ids_for([50, 10, 30, 5, 40, 15])
    h = drive_map.drive_halves(ids)
    assert set(h) == {drive_map.drive_of_log(l) for l in LOGS} and set(h.values()) == {0, 1}
    scenes = Counter()
    for i in ids:
        scenes[h[drive_map.drive_of_id(i)]] += 1
    total = len(ids)
    assert scenes[0] >= total / 2 and scenes[0] - 50 < total / 2  # half 0 stops at the first drive past half
    assert drive_map.drive_halves(list(reversed(ids))) == h  # the order of the rows does not matter
    assert any(drive_map.drive_halves(ids, seed=s) != h for s in range(1, 6))


def test_drive_of_id_takes_the_log_part():
    assert drive_map.drive_of_id(f"openscene/{LOGS[0]}/abc") == "2021.05.12.19.36.12_veh-35"
    with pytest.raises(ValueError):
        drive_map.drive_of_id("openscene/not-a-log/abc")


def test_training_on_one_drive_half(tmp_path):
    pytest.importorskip("torch")
    import torch

    from test_head_train import make_split, run

    cache = tmp_path / "cache"
    make_split(cache / "train", 24, seed=0)
    make_split(cache / "dev", 8, seed=1)
    meta = cache / "train" / "meta.jsonl"
    rows = [json.loads(l) for l in meta.read_text().splitlines()]
    counts = [8, 4, 6, 6]  # 4 drives
    for r, i in zip(rows, ids_for(counts)):
        r["id"] = i
    meta.write_text("".join(json.dumps(r) + "\n" for r in rows))
    halves = drive_map.drive_halves([r["id"] for r in rows])
    mine = [j for j, r in enumerate(rows) if halves[drive_map.drive_of_id(r["id"])] == 1]
    assert 0 < len(mine) < len(rows)
    out = tmp_path / "half1"
    log = run("train_head.py", "--cache", str(cache), "--out", str(out), "--kind", "ddim", "--width", "32",
              "--depth", "2", "--heads", "4", "--batch", "4", "--warmup", "2", "--workers", "1", "--epochs", "1",
              "--drive-half", "1")
    meta_run = json.loads((out / "run.json").read_text())
    assert meta_run["train_rows"] == len(mine) and meta_run["drive_half"]["rows"] == len(mine)
    assert meta_run["drive_half"]["half"] == 1 and meta_run["drive_half"]["drives_all"] == 4
    assert meta_run["steps_per_epoch"] == -(-len(mine) // 4)
    assert f"drive half 1: {len(mine)} rows" in log
    w = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    xy = torch.tensor([rows[j]["target_xy"] for j in mine]).reshape(len(mine), -1)
    assert torch.allclose(w["stats"]["mean"].float(), xy.mean(0), atol=1e-5)  # statistics from this half only
