"""``pilot_sft/heads/candidate_headroom.py``: own pick against the best candidate, and failures a candidate rescues."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pilot_sft" / "heads"))

import candidate_headroom as ch  # noqa: E402


def r(score, nc=1.0, dac=1.0):
    return {"score": score, "no_at_fault_collisions": nc, "drivable_area_compliance": dac}


def test_headroom(tmp_path):
    rows = [  # own pick first in "names"; scene 1: a sample beats it; scene 2: the own pick leaves the drivable area
        {"token": "t1", "names": ["single", "sample_0", "sample_1"], "results": [r(0.8), r(0.9), r(0.7)]},
        {"token": "t2", "names": ["sample_0", "single", "sample_1"], "results": [r(0.85), r(0.0, dac=0.0), {"score": 0, "error": "x"}]},
        {"token": "t3", "names": ["single", "sample_0", "sample_1"], "results": [r(0.0, nc=0.0), r(0.0, dac=0.0), r(0.6)]},
    ]
    f = tmp_path / "s.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in rows))
    h = ch.headroom(str(f))
    assert h["scenes"] == 3 and h["candidates"] == 3
    assert h["own_pick_pdms"] == round(100 * 0.8 / 3, 2) and h["best_of_k_pdms"] == round(100 * (0.9 + 0.85 + 0.6) / 3, 2)
    assert h["share_some_candidate_beats_own"] == 1.0
    assert h["own_pick_fails_nc_or_dac"] == 2 and h["share_of_failures_some_candidate_passes"] == 1.0
