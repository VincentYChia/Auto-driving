import json

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")
pytest.importorskip("PIL.Image")

from data_pipeline.v1 import audit, canonical, navsim_splits, openscene_images, pilot_view, qa_report, release, tables  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from data_pipeline.v1.openscene_index import PATH_OFFSETS_S  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402
from tests.data.test_v1_images import write_archives  # noqa: E402


@pytest.fixture
def fetched(tmp_path):
    """Steps 2–3 on three synthetic logs: index, plan, and every archive fetched."""
    logs = [v1.make_log(name, 60) for name in ("log_a", "log_b", "log_c")]
    for frames in logs:
        for frame in frames:
            frame["lidar2ego_rotation"] = [1.0, 0.0, 0.0, 0.0]
            frame["lidar2ego_translation"] = np.zeros(3)
    metadata = v1.write_logs_dir(tmp_path / "logs", logs)
    root = v1.write_navsim_root(tmp_path / "navsim", train=["log_a", "log_b"], val=["log_c"])
    out = tmp_path / "out"
    oi.build_index(str(metadata), navsim_splits.load(str(root)), str(out))
    audit.main(["--out", str(out)])
    src, archive_map, archive_tree = write_archives(tmp_path, out)
    openscene_images.plan(str(out), archive_map, archive_tree)
    openscene_images.fetch(str(out), source_dir=str(src))
    return out, metadata


def test_canonical_record_holds_every_selected_frame_once_with_its_image(fetched):
    out, metadata = fetched
    report = canonical.build(str(out), ("smoke", "v1-pilot"), str(metadata))
    assert [c["check"] for c in report["checks"] if not c["ok"]] == []
    smoke = canonical.set_tokens(str(out), "smoke")
    pilot = canonical.set_tokens(str(out), "v1-pilot")
    frames = tables.read(str(out / "frames.parquet"))
    assert sorted(frames["token"]) == sorted(set(smoke) | set(pilot)) and report["frames"] == len(frames["token"])
    assert frames["in_smoke"].sum() == len(smoke) and frames["in_v1_pilot"].sum() == len(pilot)
    k = 0
    assert frames["image_path"][k] == f"images/CAM_F0/{frames['log_name'][k]}/{frames['token'][k]}.jpg"
    assert (out / frames["image_path"][k]).exists()
    assert frames["path_xyyaw"].shape[1:] == (len(PATH_OFFSETS_S), 3) and frames["K"].shape[1:] == (4,)
    assert [p.split("/")[1] for p in frames["other_cam_paths"][k]] == list(canonical.OTHER_CAMERAS)
    assert set(frames["our_split"][frames["log_name"] == "log_c"]) == {"dev"}

    privileged = tables.read(str(out / "privileged.parquet"))
    assert list(privileged["frame_id"]) == list(frames["frame_id"])
    assert "gt_boxes" not in frames and "track_tokens" not in frames  # kept apart (R2)
    assert privileged["n_boxes"][k] == len(privileged["gt_boxes"][k]) == len(privileged["gt_names"][k])
    assert list(privileged["tl_is_red"][k]) == [False] and list(privileged["roadblock_ids"][k]) == ["rb1"]
    decisions = tables.read(str(out / "labels" / canonical.LABELS_FILE))
    assert not decisions["calibrated"].any() and list(decisions["frame_id"]) == list(frames["frame_id"])
    saved = json.loads((out / "canonical.json").read_text())
    assert saved["sets"]["smoke"]["missing_image"] == 0 and "frames.parquet" in saved["outputs"]


def test_canonical_catches_changed_images_and_leaves_out_unfetched_frames(fetched):
    out, metadata = fetched
    images = sorted((out / "images" / "CAM_F0").rglob("*.jpg"))
    images[0].write_bytes(images[0].read_bytes() + b"x")
    images[1].unlink()
    report = canonical.build(str(out), ("smoke",), str(metadata))
    failed = [c for c in report["checks"] if not c["ok"]]
    assert [c["check"] for c in failed] == ["every image file matches its step-3 SHA-256"] and failed[0]["value"] == 2

    full = report["frames"]
    (out / "images" / "archives" / "1.json").unlink()  # as if archive 1 (log_b, log_c) were not fetched yet
    report = canonical.build(str(out), ("smoke",), str(metadata), verify_images=False)
    frames = tables.read(str(out / "frames.parquet"), ["log_name"])
    assert set(frames["log_name"]) == {"log_a"} and report["sets"]["smoke"]["missing_image"] == full - report["frames"]


def test_hfov_through_the_lens_model():
    k = [[1545.0, 0, 960.0], [0, 1545.0, 560.0], [0, 0, 1]]
    pinhole = np.degrees(2 * np.arctan(960 / 1545))
    assert canonical.hfov_undistorted_deg(k, [0, 0, 0, 0, 0], 1920) == pytest.approx(pinhole, abs=0.01)
    assert canonical.hfov_undistorted_deg(k, [-0.356, 0.1725, -0.0021, 0.0005, -0.0523], 1920) > pinhole + 5  # barrel


def test_qa_report_projects_the_path_and_writes_aggregates_only(fetched):
    out, metadata = fetched
    canonical.build(str(out), ("smoke",), str(metadata))
    summary = qa_report.build(str(out), "smoke", n=3)
    assert summary["renders"]["n"] == 3 and summary["canonical_checks_failed"] == []
    page = (out / "qa" / "report.html").read_text(encoding="utf-8")
    assert page.count("data:image/jpeg;base64,") == 3
    saved = (out / "qa" / "summary.json").read_text()
    assert "base64" not in saved and "log_a_tok" not in saved  # no per-frame data in the committed file


def test_pilot_view_writes_prompts_targets_and_a_chat_export(fetched):
    out, metadata = fetched
    canonical.build(str(out), ("smoke",), str(metadata))
    spec = pilot_view.build(str(out), "smoke", width=16, workers=1)
    view = out / "views" / "pilot_v1" / "smoke"
    items = [json.loads(line) for line in (view / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(items) == spec["samples"]["train"] > 0
    item = items[0]
    assert (view / item["image"]).exists() and len(item["target_xy"]) == 8
    assert np.allclose(pilot_view.parse_path(item["target"]), item["target_xy"], atol=0.051)
    assert "Route command: go straight." in item["prompt"] and "-0.0" not in item["prompt"] + item["target"]
    assert all(s not in item["prompt"] for s in ("+0.5 s", "decision", "gt_"))  # no future or privileged data
    assert spec["constant_velocity_baseline"]["train"]["samples"] == len(items)
    exported = pilot_view.export(str(out), "smoke", limit=2)
    record = json.loads((view / "chat_train.jsonl").read_text().splitlines()[0])
    assert exported["train"]["samples"] == 2 and [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert pilot_view.parse_path("(1.0, 2.0) (3, -4)") is None


def test_release_pins_tables_and_images_and_refuses_an_incomplete_set(fetched):
    out, metadata = fetched
    canonical.build(str(out), ("smoke", "v1-pilot"), str(metadata))
    manifest = release.build(str(out), "v1-pilot")
    sums = (out / "images" / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == manifest["images"]["count"] and "frames.parquet" in manifest["tables"]
    card = (out / "DATA_CARD.md").read_text(encoding="utf-8")
    assert "Gate 2" in card and "log_a_tok" not in card and "log_a_tok" not in (out / "manifest.json").read_text()
    (out / "images" / "archives" / "1.json").unlink()
    canonical.build(str(out), ("smoke", "v1-pilot"), str(metadata), verify_images=False)
    with pytest.raises(SystemExit):
        release.build(str(out), "v1-pilot")
