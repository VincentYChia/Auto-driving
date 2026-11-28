"""``pilot_sft/heads/infold_vs_xfit.py``: in-fold against cross-fitted selector picks on the same navdev scenes."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "pilot_sft" / "heads"))

import infold_vs_xfit  # noqa: E402

TOKENS = [f"tok{i}" for i in range(12)]
DRIVES = {t: f"drive{i % 4}" for i, t in enumerate(TOKENS)}


def write(heads, x, variant, true, chosen=0):
    d = heads / f"selector-{x}-{variant}" / "navdev"
    d.mkdir(parents=True)
    rows = [{"token": t, "chosen": chosen, "true_chosen": true + 0.01 * (i % 2)} for i, t in enumerate(TOKENS)]
    rows.append({"token": "unscored-dev-scene", "chosen": 3})  # outside NAVSIM's navdev filter: no true score
    (d / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_in_fold_minus_cross_fitted_per_contender_and_pooled_over_rl(tmp_path):
    write(tmp_path, "d6", "infold-s0", 0.84, chosen=2)
    write(tmp_path, "d6", "xfit-s0", 0.80, chosen=1)
    write(tmp_path, "a4", "infold-s0", 0.88)
    write(tmp_path, "a4", "xfit-s0", 0.88)
    write(tmp_path, "d4", "infold-s0", 0.82)
    write(tmp_path, "d4", "xfit-s0", 0.83)
    write(tmp_path, "c6", "xfit-s0", 0.87)  # no in-fold file yet: left out
    r = infold_vs_xfit.compare(str(tmp_path), DRIVES)
    assert set(r["contenders"]) == {"A4", "D4", "D6"}
    d6 = r["contenders"]["D6"]
    assert d6["in_fold_minus_cross_fitted"] == 4.0 and d6["ci_drives"] == [4.0, 4.0] and d6["scenes"] == 12
    assert d6["same_pick"] == 0.0 and (d6["in_fold"], d6["cross_fitted"]) == (84.5, 80.5) and d6["rl"] is True
    assert r["contenders"]["A4"]["same_pick"] == 1.0 and r["contenders"]["D4"]["rl"] is False
    pooled = r["pooled_with_rl"]
    assert pooled["contenders"] == ["A4", "D6"] and pooled["in_fold_minus_cross_fitted"] == 2.0  # D4 is not pooled
    out = tmp_path / "out.json"
    (tmp_path / "drives.json").write_text(json.dumps(DRIVES))
    infold_vs_xfit.main(["--heads", str(tmp_path), "--drives", str(tmp_path / "drives.json"), "--out", str(out)])
    assert json.loads(out.read_text())["contenders"]["D6"]["in_fold_minus_cross_fitted"] == 4.0
