"""``verify_v1_pool.py`` (PRETRAIN_V1_PLAN.md §11) on the synthetic v1.0-pool plan of test_v1_pool_plan.py."""

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import scene_exclusion  # noqa: E402
from tests.data import v1_synthetic  # noqa: E402
from tests.data.test_v1_pool_plan import LOGS, plan_pool, pool_inputs  # noqa: E402,F401
from tests.data.test_v1_xl_plan import sha, tokens  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERIFY = os.path.join(ROOT, "scripts", "anvil", "data_pipeline", "verify_v1_pool.py")


def load():
    spec = importlib.util.spec_from_file_location("verify_v1_pool", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failed(checks):
    return [c["check"] for c in checks if not c["ok"]]


@pytest.fixture
def planned(pool_inputs, monkeypatch):
    """The synthetic plan, with its warm-up list known (so the manifest records a source, as the real one does)."""
    monkeypatch.setitem(scene_exclusion.KNOWN_LISTS, sha(Path(pool_inputs["warm"])), {"filter": "warmup_test_e2e"})
    manifest = plan_pool(pool_inputs)
    return {**pool_inputs, "manifest": manifest, "out_dir": str(pool_inputs["out"])}


def rewrite(path, keep=lambda t: True, extra=()):
    scene_exclusion.write_tokens(path, [t for t in tokens(path) if keep(t)] + list(extra))


def test_rules_recompute_the_four_sets(planned):
    v = load()
    checks = v.check_rules(planned["out_dir"], planned["meta"])
    assert not failed(checks)
    reasons = next(c for c in checks if "exactly one reason" in c["check"])["value"]
    assert reasons["first_failing_rule"] == {"index_row": 3, "train_log": 5, "eligible": 1, "outside_window": 5}
    assert reasons["tokens_failing_two_or_more_rules"] == 1  # dev_lv tok20: a dev row inside a warm-up window
    no_metadata = failed(v.check_rules(planned["out_dir"], None))
    assert len(no_metadata) == 1 and "NOT placed" in no_metadata[0]


def test_rules_catch_a_set_that_breaks_its_rule(planned):
    v = load()
    s = planned["out"] / "splits"
    mix, nt = s / "v1-xl-mix_tokens_train.txt", s / "v1-nt_tokens_train.txt"
    original = mix.read_bytes()
    rewrite(mix, extra=["lv_keep_tok30"])  # a Las Vegas keep-going pool row: the rule does not add it
    assert any("holds nothing else" in n for n in failed(v.check_rules(planned["out_dir"], planned["meta"])))
    mix.write_bytes(original)
    rewrite(mix, keep=lambda t: t != "bos_side_tok51")  # a Boston row the rule adds, left out
    assert any("outside Las Vegas" in n for n in failed(v.check_rules(planned["out_dir"], planned["meta"])))
    mix.write_bytes(original)
    rewrite(mix, keep=lambda t: not t.startswith("lv_stop"), extra=["lv_stop_tok03", "lv_stop_tok13"])  # 2 stay rows
    assert any("stay-stopped rows" in n for n in failed(v.check_rules(planned["out_dir"], planned["meta"])))
    mix.write_bytes(original)
    rewrite(nt, keep=lambda t: t != "lv_side_tok10")
    names = failed(v.check_rules(planned["out_dir"], planned["meta"]))
    assert any(n.startswith("v1-nt = the navtrain rows") for n in names), names


def devkit(where, navtrain, filters):
    return str(v1_synthetic.write_navsim_root(where, train=[log for log in LOGS if log != "dev_lv"], val=["dev_lv"],
                                              navtrain_logs=list(LOGS), navtrain_tokens=navtrain, filters=filters))


def test_leakage_against_two_devkits_states_the_scene_level_exemption(planned, tmp_path):
    v = load()
    navtrain = tokens(planned["out"] / "splits" / scene_exclusion.NAVTRAIN_FILE)
    warm = tokens(planned["out"] / "splits" / scene_exclusion.WARMUP_FILE)
    v11 = devkit(tmp_path / "navsim-v1.1", navtrain, {
        "warmup_test_e2e": {"log_names": ["bos_side", "dev_lv", "lv_keep"], "tokens": warm},
        "navtest": {"log_names": [], "tokens": ["nothing_selected"]}})
    main = devkit(tmp_path / "navsim", navtrain, {"navtest": {"log_names": [], "tokens": ["nothing_selected"]}})
    checks = v.check_leakage(planned["out_dir"], [v11, main], planned["meta"])
    assert not failed(checks), failed(checks)
    assert any(c["check"].endswith("[navsim-v1.1]") for c in checks) and any(c["check"].endswith("[navsim]") for c in checks)
    exempt = next(c for c in checks if "is held to the scene-level rule" in c["check"])
    assert "SCENE_LEVEL_FILTERS" in exempt["check"] and exempt["value"]["scene_level_logs"] == 3
    assert exempt["value"]["train_rows_in_them"]["v1-all"] > 0  # rows of bos_side and lv_keep outside the windows
    other = next(c for c in checks if "other guarded lists" in c["check"])
    assert "every list but the scene-level warmup_test_e2e" in other["check"]
    assert other["value"]["exempt_scene_level"] == ["warmup_test_e2e"]
    # a navtrain that is not the committed list, and a guarded token in a train set, fail
    other_navtrain = devkit(tmp_path / "navsim-other", navtrain[1:], {})
    names = failed(v.check_leakage(planned["out_dir"], [other_navtrain], planned["meta"]))
    assert any("navtrain tokens = the committed list" in n for n in names), names
    guarded = devkit(tmp_path / "navsim-guard", navtrain, {"private_test_e2e": {"log_names": [], "tokens": ["lv_side_tok10"]}})
    assert any("other guarded lists" in n for n in failed(v.check_leakage(planned["out_dir"], [guarded], planned["meta"])))


def test_splits_on_the_synthetic_plan(planned, monkeypatch):
    v = load()
    s = planned["out"] / "splits"
    expected = {name: {"train": len(tokens(s / f"{name}_tokens_train.txt")), "dev": 3,
                       "train_sha256": sha(s / f"{name}_tokens_train.txt")} for name in v.SETS}
    monkeypatch.setattr(v.xl, "DEV_SHA256", sha(s / "v1-all_tokens_dev.txt"))
    monkeypatch.setattr(v.xl, "WARMUP_SHA256", sha(s / scene_exclusion.WARMUP_FILE))
    monkeypatch.setattr(v.xl, "WARMUP_TOKENS", 4)
    monkeypatch.setattr(v, "BASE_TRAIN_SHA256", sha(s / "v1-xl-nd_tokens_train.txt"))
    monkeypatch.setattr(v, "BASE_TRAIN", 9)
    monkeypatch.setattr(v, "NAVTRAIN_SHA256", sha(s / scene_exclusion.NAVTRAIN_FILE))
    monkeypatch.setattr(v, "NAVTRAIN_TOKENS", len(tokens(s / scene_exclusion.NAVTRAIN_FILE)))
    build = v.POOL._replace(expected=expected, frames_to_keep=planned["manifest"]["frames_to_keep"])
    ref = os.path.dirname(os.path.dirname(planned["extends"]))  # the folder holding the base's splits/
    checks = v.xl.check_splits(planned["out_dir"], ref, build)
    assert failed(checks) == ["archive order lists 200 distinct archives"]  # the synthetic map has 2 archives
    rewrite(s / "v1-nt-mix_tokens_train.txt", keep=lambda t: t != "lv_side_tok10")  # a v1-nt row left out
    names = failed(v.xl.check_splits(planned["out_dir"], ref, build))
    assert "train: v1-nt ⊂ v1-nt-mix ⊂ v1-all and v1-xl-mix ⊂ v1-nt-mix" in names
    assert "train: v1-nt-mix = v1-nt ∪ v1-xl-mix" in names
