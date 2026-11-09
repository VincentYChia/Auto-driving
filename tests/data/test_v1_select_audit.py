import json
import os
import re

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import audit, navsim_splits, selection  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from data_pipeline.v1.selection import SelectionConfig  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402


def fake(per_log, splits):
    """Candidate columns for one log per entry of ``splits``: anchors every 0.5 s, all cruise."""
    n = per_log * len(splits)
    return {
        "log_name": np.array([f"log{k}" for k in range(len(splits)) for _ in range(per_log)], dtype=object),
        "t_us": np.tile(np.arange(per_log, dtype=np.int64) * 500_000, len(splits)),
        "our_split": np.array([s for s in splits for _ in range(per_log)], dtype=object),
        "has_cam_f0": np.ones(n, dtype=bool),
        "pose_ok": np.ones(n, dtype=bool),
        "standstill": np.zeros(n, dtype=bool),
        "decision_lon": np.full(n, "keep_speed", dtype=object),
        "decision_lat": np.full(n, "straight", dtype=object),
    }


def times_by_log(cands, rows):
    out = {}
    for r in rows:
        out.setdefault(cands["log_name"][r], []).append(int(cands["t_us"][r]))
    return {k: sorted(v) for k, v in out.items()}


def test_selection_is_spaced_split_by_log_and_seeded():
    cands = fake(200, ["train", "train", "dev"])
    config = SelectionConfig("t", n_train=60, n_dev=20, spacing_s=2.0)
    rows, report = selection.select(cands, config)
    assert report["splits"]["train"]["selected"] == 60 and report["splits"]["dev"]["selected"] == 20
    for times in times_by_log(cands, rows).values():
        assert min(np.diff(times)) >= 1.95e6
    assert set(cands["log_name"][rows][cands["our_split"][rows] == "dev"]) == {"log2"}
    again, _ = selection.select(cands, config)
    other, _ = selection.select(cands, SelectionConfig("t", n_train=60, n_dev=20, seed=1))
    assert again.tolist() == rows.tolist() and other.tolist() != rows.tolist()


def test_cruise_cap_holds_when_other_classes_suffice_and_is_reported_when_not():
    cands = fake(400, ["train"] * 5)
    cands["decision_lat"][::3] = "curve_left"
    config = SelectionConfig("t", n_train=200, n_dev=0)
    rows, report = selection.select(cands, config)
    split = report["splits"]["train"]
    assert split["selected"] == 200 and split["cruise_share"] == 0.4 and not split["cruise_cap_exceeded"]
    assert split["decision_lat"] == {"curve_left": 120, "straight": 80}

    cands["decision_lat"][:] = "straight"
    cands["decision_lat"][:10] = "curve_left"  # 10 non-cruise anchors within 5 s of one log
    _, report = selection.select(cands, config)
    split = report["splits"]["train"]
    assert split["selected"] == 200 and split["cruise_cap_exceeded"] and split["decision_lat"]["curve_left"] <= 3


def test_gap_filling_packs_close_to_the_spacing_limit():
    cands = fake(400, ["train"])  # 200 s: at most 100 anchors 2 s apart
    rows, report = selection.select(cands, SelectionConfig("t", n_train=100, n_dev=0))
    # random order alone reaches 83 (V, 2026-11-08); the fill pass then adds nothing
    assert report["splits"]["train"]["selected"] >= 80 and report["splits"]["train"]["stages"]["fill"] == 0
    assert min(np.diff(sorted(cands["t_us"][rows]))) >= 1.95e6


def test_shortfall_is_reported():
    cands = fake(20, ["train"])
    _, report = selection.select(cands, SelectionConfig("t", n_train=50, n_dev=0, spacing_s=2.0))
    split = report["splits"]["train"]
    assert 0 < split["selected"] <= 5 and split["shortfall"] == 50 - split["selected"]


def test_ineligible_rows_and_unavailable_logs_are_never_selected():
    cands = fake(100, ["train", "train", "excluded"])
    cands["pose_ok"][:50] = False
    cands["has_cam_f0"][50:60] = False
    take_all = SelectionConfig("t", n_train=1000, n_dev=0, spacing_s=0.0, cruise_cap=1.0)
    rows, report = selection.select(cands, take_all)
    assert rows.tolist() == list(range(60, 200)) and report["splits"]["train"]["eligible"] == 140
    rows, report = selection.select(cands, take_all, available_logs=["log1"])
    assert set(cands["log_name"][rows]) == {"log1"} and report["available_logs"] == 1


def test_standstill_thinning_keeps_one_anchor_per_period():
    cands = fake(40, ["train"])
    cands["standstill"][:] = True
    keep = selection.thin_standstill(cands, np.ones(40, dtype=bool), 5.0)
    assert np.flatnonzero(keep).tolist() == [0, 10, 20, 30]
    cands["standstill"][20:] = False
    keep = selection.thin_standstill(cands, np.ones(40, dtype=bool), 5.0)
    assert np.flatnonzero(keep).tolist() == [0, 10] + list(range(20, 40))


def test_balanced_draw_favours_rare_city_and_manoeuvre_cells_in_train_only():
    cands = fake(400, ["train"] * 10 + ["dev"] * 10)
    cands["city"] = np.where(np.isin(cands["log_name"], ["log0", "log10"]), "sg-one-north", "us-nv-las-vegas-strip").astype(object)
    natural = SelectionConfig("t", n_train=200, n_dev=200, cruise_cap=1.0)
    balanced = SelectionConfig("t", n_train=200, n_dev=200, cruise_cap=1.0, balance_alpha=1.0)
    _, plain = selection.select(cands, natural)
    _, weighted = selection.select(cands, balanced)
    share = {name: r["splits"]["train"]["city"].get("sg-one-north", 0) / 200 for name, r in (("plain", plain), ("weighted", weighted))}
    # 1 log in 10, weighted 9x per anchor; its 200 s fit ~75 anchors 2 s apart, so the spacing caps it
    assert share["plain"] < 0.15 and share["weighted"] > 0.25
    assert weighted["splits"]["dev"] == plain["splits"]["dev"]  # dev keeps the natural mix
    again, _ = selection.select(cands, balanced)
    assert again.tolist() == selection.select(cands, balanced)[0].tolist()


def test_fixed_none_and_no_fixed_rows_select_the_same_and_report_the_stages():
    cands = fake(400, ["train"] * 4 + ["dev"])
    cands["decision_lat"][::3] = "curve_left"
    cands["city"] = np.where(cands["log_name"] == "log0", "sg-one-north", "us-nv-las-vegas-strip").astype(object)
    config = SelectionConfig("t", n_train=150, n_dev=30, spacing_s=2.0, balance_alpha=0.5)
    rows, report = selection.select(cands, config)
    for fixed in ([], np.zeros(0, dtype=np.int64), {"train": [], "dev": []}):
        again, other = selection.select(cands, config, fixed=fixed)
        assert again.tolist() == rows.tolist() and other == report
    split = report["splits"]["train"]
    assert split["fixed"] == 0 and split["added"] == split["selected"] == 150
    assert sum(split["stages"].values()) == 150 and split["stages"]["others_1"] == 150 - int(0.4 * 150)


def test_fixed_rows_that_are_the_whole_draw_reproduce_it():
    cands = fake(400, ["train"] * 3 + ["dev"])
    cands["city"] = np.full(len(cands["t_us"]), "us-ma-boston", dtype=object)
    config = SelectionConfig("t", n_train=120, n_dev=20, spacing_s=2.0, balance_alpha=0.5)
    rows, _ = selection.select(cands, config)
    again, report = selection.select(cands, config, fixed=rows)
    assert again.tolist() == rows.tolist() and report["splits"]["train"]["added"] == 0


def test_a_superset_at_closer_spacing_keeps_every_fixed_row_and_the_dev_set():
    cands = fake(400, ["train"] * 4 + ["dev"] * 2)
    cands["decision_lat"][::4] = "curve_left"
    cands["city"] = np.where(cands["log_name"] == "log0", "sg-one-north", "us-nv-las-vegas-strip").astype(object)
    small = SelectionConfig("small", n_train=200, n_dev=40, spacing_s=2.0, balance_alpha=0.5)
    rows, _ = selection.select(cands, small)
    split = cands["our_split"][rows]
    fixed = {"train": rows[split == "train"], "dev": rows[split == "dev"]}
    big = SelectionConfig("big", n_train=400, n_dev=40, spacing_s=1.0, balance_alpha=0.5)
    sup, report = selection.select(cands, big, fixed=fixed)
    train = report["splits"]["train"]
    assert np.isin(rows, sup).all() and train["shortfall"] == 0 and train["fixed"] == 200 and train["added"] == 200
    assert not train["cruise_cap_exceeded"] and sum(train["stages"].values()) == 200  # the quota counts fixed rows
    dev = sup[cands["our_split"][sup] == "dev"]
    assert dev.tolist() == fixed["dev"].tolist() and report["splits"]["dev"]["added"] == 0
    for times in times_by_log(cands, sup).values():
        assert min(np.diff(times)) >= 0.95e6


@pytest.mark.parametrize("n_dev, added", [(20, 0), (24, 4)])
def test_cruise_heavy_fixed_rows_never_push_a_split_past_its_target(n_dev, added):
    cands = fake(400, ["train", "train", "dev"])
    cands["decision_lat"][1::2] = "curve_left"  # odd rows other classes, even rows cruise
    dev_rows = np.flatnonzero(cands["our_split"] == "dev")
    fixed_dev = dev_rows[::4][:20]  # 20 cruise rows, 2 s apart: more than 40 % cruise
    rows, report = selection.select(cands, SelectionConfig("t", n_train=40, n_dev=n_dev, spacing_s=1.0),
                                    fixed={"dev": fixed_dev})
    dev = report["splits"]["dev"]
    assert dev["selected"] == n_dev and dev["shortfall"] == 0 and dev["added"] == added
    assert set(fixed_dev) <= set(rows.tolist())


def test_fixed_rows_that_break_a_rule_raise():
    cands = fake(100, ["train", "dev"])
    config = SelectionConfig("t", n_train=40, n_dev=10, spacing_s=2.0)
    with pytest.raises(RuntimeError, match="closer than"):
        selection.select(cands, config, fixed=[0, 2])  # 1.0 s apart in log0
    cands["pose_ok"][5] = False
    with pytest.raises(RuntimeError, match="rule 2, 3 or 5"):
        selection.select(cands, config, fixed=[5])
    with pytest.raises(RuntimeError, match="unavailable"):
        selection.select(cands, config, available_logs=["log1"], fixed=[10])
    cands["standstill"][:20] = True  # thinning keeps rows 0 and 10 of log0
    with pytest.raises(RuntimeError, match="standstill thinning"):
        selection.select(cands, config, fixed=[0, 4])
    with pytest.raises(RuntimeError, match="not in the index's dev split"):
        selection.select(cands, config, fixed={"dev": [30]})  # row 30 is in log0, a train log
    with pytest.raises(ValueError, match="twice"):
        selection.select(cands, config, fixed=[30, 30])
    with pytest.raises(RuntimeError, match="exceed the target"):
        selection.select(cands, SelectionConfig("t", n_train=1, n_dev=0, spacing_s=2.0), fixed=[30, 40])


def test_check_raises_when_a_fixed_row_is_not_selected():
    cands = fake(20, ["train"])
    base = selection.eligible(cands)
    codes = selection.log_codes(cands)
    selection._check(cands, np.array([0, 4]), base, codes, 2.0, fixed=np.array([4]))
    with pytest.raises(RuntimeError, match="fixed row"):
        selection._check(cands, np.array([0, 4]), base, codes, 2.0, fixed=np.array([8]))


REAL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pretrain")
REAL_INDEX = os.path.join(REAL, "v1.0-large", "index", "candidates.parquet")


@pytest.fixture(scope="module")
def real():
    """The real index and the committed token files (local only; skipped where the index is absent)."""
    if not os.path.exists(REAL_INDEX):
        pytest.skip("the real candidate index is not here")
    from data_pipeline.v1 import openscene_images, tables

    cands = tables.read(REAL_INDEX, openscene_images.PLAN_COLUMNS)
    with open(os.path.join(REAL, "v1.0-large", "images", "archives.json"), encoding="utf-8") as f:
        archives = json.load(f)
    with open(os.path.join(REAL, "v1.0-pilot", "splits", "manifest.json"), encoding="utf-8") as f:
        smoke_archives = json.load(f)["smoke_archives"]
    return cands, archives, smoke_archives


@pytest.mark.parametrize("config, folder, logs", [(selection.PILOT, "v1.0-pilot", "all"),
                                                  (selection.LARGE, "v1.0-large", "all"),
                                                  (selection.SMOKE, "v1.0-pilot", "smoke")])
def test_fixed_none_reproduces_the_committed_token_files(real, config, folder, logs):
    cands, archives, smoke_archives = real
    numbers = list(archives) if logs == "all" else [str(n) for n in smoke_archives]
    rows, _ = selection.select(cands, config, available_logs=[lg for n in numbers for lg in archives[n]["logs"]],
                               fixed=None)
    for split in ("train", "dev"):
        with open(os.path.join(REAL, folder, "splits", f"{config.name}_tokens_{split}.txt"), encoding="utf-8") as f:
            committed = f.read().split()
        assert sorted(cands["token"][rows][cands["our_split"][rows] == split].tolist()) == committed


def test_simulate_uses_an_archive_sized_share_of_logs():
    cands = fake(50, ["train"] * 19 + ["dev"])
    report = selection.simulate(cands, SelectionConfig("t", 10, 2), sorted(set(cands["log_name"])), n_archives=20)
    assert report["available_logs"] == 2 and report["simulated_archives"] == 20


def _strict_json(text):
    def refuse(constant):
        raise ValueError(f"non-standard JSON constant {constant}")

    return json.loads(text, parse_constant=refuse)


def test_audit_writes_aggregate_reports_and_splits(tmp_path):
    logs = [v1.make_log("log_a", 40), v1.make_log("log_b", 40), v1.make_log("log_t", 30), v1.make_standstill(v1.make_log("log_s", 40))]
    root = v1.write_navsim_root(tmp_path / "navsim", train=["log_a", "log_s"], val=["log_b"], test=["log_t"], navtrain_tokens=["log_a_tok05"])
    out = tmp_path / "out"
    oi.build_index(str(v1.write_logs_dir(tmp_path / "logs", logs)), navsim_splits.load(str(root)), str(out))
    audit.main(["--out", str(out)])

    summary = _strict_json((out / "audit" / "summary.json").read_text())
    assert summary["logs"]["total"] == 4 and summary["logs"]["by_our_split"] == {"train": 2, "dev": 1, "excluded": 1}
    assert summary["frames"]["total"] == 150 and summary["frames"]["valid_windows (rule 1)"] == 29 * 3 + 19
    assert summary["rules"]["5 not guarded"] == 29 * 3
    assert summary["rules"]["6 after standstill thinning"] == 29 * 2 + 3  # log_s: anchors at 1.5, 6.5, 11.5 s
    assert summary["distributions"]["train"]["decision_lon"] == {"keep_speed": 29, "stopped": 29}
    assert summary["command_vs_path"]["straight"]["rows"] == 29 * 3
    checks = {c["check"]: c["ok"] for c in summary["checks"]}
    assert not checks["no nuPlan test or NAVSIM evaluation log in the trainval metadata"]  # log_t was planted
    assert checks["every navtrain token found has a valid window"] and checks["frame ids are unique"]

    report = (out / "audit" / "report.md").read_text()
    assert "## Decisions for Gate 0" in report and "Partial run" not in report
    assert (out / "splits" / "train_logs.txt").read_text() == "log_a\nlog_s\n"
    assert (out / "splits" / "dev_logs.txt").read_text() == "log_b\n"
    assert (out / "splits" / "excluded_logs.txt").read_text() == "log_t\tnuplan_test\n"
    for text in (report, (out / "audit" / "summary.json").read_text()):  # aggregate only: no frame ids
        assert not re.search(r"log_[a-z]_tok\d\d", text) and "CAM_F0/" not in text


def test_audit_marks_a_partial_run(tmp_path):
    root = v1.write_navsim_root(tmp_path / "navsim", train=["log_a", "log_b"])
    out = tmp_path / "out"
    logs_dir = v1.write_logs_dir(tmp_path / "logs", [v1.make_log("log_a"), v1.make_log("log_b")])
    oi.build_index(str(logs_dir), navsim_splits.load(str(root)), str(out), max_logs=1)
    audit.main(["--out", str(out)])
    assert "Partial run" in (out / "audit" / "report.md").read_text()
