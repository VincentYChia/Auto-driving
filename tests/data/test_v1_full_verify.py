"""``verify_v1_full.py`` (PRETRAIN_V1_PLAN.md §12) on the synthetic v1.0-full plan of test_v1_full_plan.py."""

import hashlib
import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import scene_exclusion  # noqa: E402
from tests.data import v1_synthetic  # noqa: E402
from tests.data.test_v1_full_plan import TRAIN_LOGS, VAL_LOGS, full_inputs, plan_full  # noqa: E402,F401
from tests.data.test_v1_xl_plan import sha, tokens  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VERIFY = os.path.join(ROOT, "scripts", "anvil", "data_pipeline", "verify_v1_full.py")


def load():
    spec = importlib.util.spec_from_file_location("verify_v1_full", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def failed(checks):
    return [c["check"] for c in checks if not c["ok"]]


@pytest.fixture
def planned(full_inputs, monkeypatch):
    """The synthetic plan, with its warm-up list known (so the manifest records a source, as the real one does)."""
    monkeypatch.setitem(scene_exclusion.KNOWN_LISTS, sha(Path(full_inputs["warm"])), {"filter": "warmup_test_e2e"})
    manifest = plan_full(full_inputs)
    return {**full_inputs, "manifest": manifest, "out_dir": str(full_inputs["out"]), "s": full_inputs["out"] / "splits"}


def rewrite(path, keep=lambda t: True, extra=()):
    scene_exclusion.write_tokens(path, [t for t in tokens(path) if keep(t)] + list(extra))


def test_rules_recompute_the_sets_the_held_out_logs_and_dev_v2(planned):
    v = load()
    out, meta, s = planned["out_dir"], planned["meta"], planned["s"]
    assert not failed(v.check_rules(out, meta))
    no_metadata = failed(v.check_rules(out, None))
    assert len(no_metadata) == 1 and "NOT placed" in no_metadata[0]
    train, dev, held = s / "v1-full_tokens_train.txt", s / "v1-full_tokens_dev.txt", s / "dev-v2_logs.txt"
    originals = {p: p.read_bytes() for p in (train, dev, held)}
    rewrite(train, keep=lambda t: not t.startswith(VAL_LOGS[4]))  # a released val log (D): its rows left out
    assert any(n.startswith("v1-full train = every eligible") for n in failed(v.check_rules(out, meta)))
    train.write_bytes(originals[train])
    first, other = tokens(dev)[0], next(t for t in tokens(s / "v1-full-devlogs_tokens_dev.txt") if t not in tokens(dev))
    rewrite(dev, keep=lambda t: t != first, extra=[other])  # another held-out row: the draw does not give it
    assert any("reproduces dev-v2" in n for n in failed(v.check_rules(out, meta)))
    dev.write_bytes(originals[dev])
    held.write_bytes("".join(f"{log}\n" for log in tokens(held)[1:]).encode())  # half of drive A
    names = failed(v.check_rules(out, meta))
    assert any("whole nuPlan val drives" in n for n in names) and any("choose_dev_logs picks" in n for n in names)


def test_splits_on_the_synthetic_plan(planned, monkeypatch):
    v = load()
    s, m = planned["s"], planned["manifest"]
    expected = {"v1-full": {"train": len(tokens(s / "v1-full_tokens_train.txt")), "dev": 9,
                            "train_sha256": sha(s / "v1-full_tokens_train.txt"),
                            "dev_sha256": sha(s / "v1-full_tokens_dev.txt")},
                "v1-full-devlogs": {"train": 0, "dev": len(tokens(s / "v1-full-devlogs_tokens_dev.txt")),
                                    "train_sha256": v.EMPTY_SHA256, "dev_sha256": sha(s / "v1-full-devlogs_tokens_dev.txt")}}
    monkeypatch.setattr(v.xl, "DEV_SHA256", sha(s / "dev-v1_tokens.txt"))
    monkeypatch.setattr(v.xl, "WARMUP_SHA256", sha(s / scene_exclusion.WARMUP_FILE))
    monkeypatch.setattr(v.xl, "WARMUP_TOKENS", 3)
    monkeypatch.setattr(v, "BASE_TRAIN_SHA256", sha(s / "v1-all_tokens_train.txt"))
    monkeypatch.setattr(v, "BASE_TRAIN", 3)
    monkeypatch.setattr(v, "DEV_V2_LOGS", 3)
    monkeypatch.setattr(v, "DEV_V2_LOGS_SHA256", sha(s / "dev-v2_logs.txt"))
    monkeypatch.setattr(v, "SHARED_FRAMES", m["covered_by_base"]["frames"])
    build = v.FULL_BUILD._replace(expected=expected, frames_to_keep=m["frames_to_keep"])
    ref = os.path.dirname(os.path.dirname(planned["extends"]))  # the folder holding the base's splits/
    checks = v.xl.check_splits(planned["out_dir"], ref, build)
    assert failed(checks) == ["archive order lists 200 distinct archives"]  # the synthetic map has 2 archives
    assert not any(c["check"].startswith("dev is the same 2,000") for c in checks)  # no set has the old pinned dev
    rewrite(s / "v1-full_tokens_train.txt", keep=lambda t: t != f"{TRAIN_LOGS[0]}_tok40")  # a v1-all row left out
    names = failed(v.xl.check_splits(planned["out_dir"], ref, build))
    assert "every v1-all train row is in v1-full's train (every train frame v1.0-pool extracted is extracted again)" in names
    (s / "dev-v1_tokens.txt").write_text("")
    assert any(n.startswith("dev-v1_tokens.txt: the old pinned dev") for n in failed(v.xl.check_splits(planned["out_dir"], ref, build)))


def devkit(where, filters):
    return str(v1_synthetic.write_navsim_root(where, train=list(TRAIN_LOGS), val=list(VAL_LOGS),
                                              navtrain_logs=list(TRAIN_LOGS + VAL_LOGS), navtrain_tokens=["x"],
                                              filters=filters))


def test_leakage_against_two_devkits_keeps_the_held_out_logs_out_of_training(planned, tmp_path):
    v = load()
    out, meta, s = planned["out_dir"], planned["meta"], planned["s"]
    warm = tokens(s / scene_exclusion.WARMUP_FILE)
    v11 = devkit(tmp_path / "navsim-v1.1", {"warmup_test_e2e": {"log_names": [TRAIN_LOGS[0], VAL_LOGS[2]], "tokens": warm},
                                             "navtest": {"log_names": [], "tokens": ["nothing_selected"]}})
    main = devkit(tmp_path / "navsim", {"navtest": {"log_names": [], "tokens": ["nothing_selected"]}})
    checks = v.check_leakage(out, [v11, main], meta)
    assert not failed(checks), failed(checks)
    released = next(c for c in checks if "released by the re-split" in c["check"])
    assert released["value"]["train_tokens_in_released_val_logs"] > 0
    apart = next(c for c in checks if "in no train set" in c["check"])
    assert apart["value"]["held_out_logs"] == 3 and apart["value"]["held_out_drives"] == 2
    dev = tokens(s / "v1-full_tokens_dev.txt")
    guarded = devkit(tmp_path / "navsim-guard", {"private_test_e2e": {"log_names": [], "tokens": [dev[0]]}})
    assert any("other guarded lists" in n for n in failed(v.check_leakage(out, [guarded], meta)))
    held_row = next(t for t in tokens(s / "v1-full-devlogs_tokens_dev.txt") if t not in dev)
    rewrite(s / "v1-full_tokens_train.txt", extra=[held_row])  # a held-out row in training
    names = failed(v.check_leakage(out, [], meta))
    assert any("in no train set" in n for n in names), names
    assert any(n.startswith("v1-full-devlogs has no train rows and no token of v1-full's train")
               for n in failed(v._relations(out, *splits_of(v, out), v.xl.manifest(out))))


def splits_of(v, out):
    sel = v.xl.selected(out, v.FULL_BUILD)
    return ({n: {t for t, s in sel[n].items() if s == "train"} for n in v.SETS},
            {n: {t for t, s in sel[n].items() if s == "dev"} for n in v.SETS})


# --- views ---------------------------------------------------------------------------------------


def jpeg():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 36), (120, 60, 30)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def write_zip(path, members):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(members):
            z.writestr(name, members[name])


def write_view(folder, out, version, image, split_files, prompt="Past positions", tweak=None):
    """A view as pilot_view writes it, for the tokens of the set's files; ``image`` is every member's bytes."""
    folder.mkdir(parents=True, exist_ok=True)
    members = {}
    for split, path in split_files.items():
        lines = []
        for t in tokens(path):
            log = t.rsplit("_tok", 1)[0]
            item = {"id": f"{log}/{t}", "image": f"images/{log}/{t}.jpg", "system": "s", "prompt": prompt,
                    "target": "(1.0, 0.0)", "target_xy": [[1.0, 0.0]], "target_yaw": [0.0],
                    "meta": {"split": split, "city": "c", "command": "straight", "speed": 1.0}}
            if tweak:
                tweak(item)
            lines.append(json.dumps(item))
            members[item["image"]] = image
        (folder / f"{split}.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    write_zip(folder / "images.zip", members)
    data = (folder / "images.zip").read_bytes()
    (folder / "view.json").write_text(json.dumps({
        "storage": "zip", "set": "v1-full", "prompt_version": version,
        "images_zip": {"members": len(members), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}))


@pytest.fixture
def views(planned):
    """Packs and records of every frame, and both views of v1-full, consistent with each other."""
    from data_pipeline.v1 import pilot_view

    out, s = planned["out"], planned["s"]
    original = jpeg()
    every = sorted(set(tokens(s / "v1-full_tokens_train.txt")) | set(tokens(s / "v1-full-devlogs_tokens_dev.txt")))
    (out / "images" / "archives").mkdir(parents=True, exist_ok=True)
    (out / "images" / "packs").mkdir(parents=True, exist_ok=True)
    images = [{"token": t, "log": t.rsplit("_tok", 1)[0], "sha256": hashlib.sha256(original).hexdigest(),
               "bytes": len(original)} for t in every]
    (out / "images" / "archives" / "0.json").write_text(json.dumps({"archive": 0, "images": images}))
    write_zip(out / "images" / "packs" / "0.zip", {f"images/CAM_F0/{i['log']}/{i['token']}.jpg": original for i in images})
    small = pilot_view.resize_bytes(original)
    files = {"train": s / "v1-full_tokens_train.txt", "dev": s / "v1-full_tokens_dev.txt"}
    write_view(out / "views" / "pilot_v1" / "v1-full", out, "pilot_v1.1", small, files)
    write_view(out / "views" / "pilot_v1.2" / "v1-full", out, "pilot_v1.2", small, files, prompt="Past poses (x, y, h)")
    pool_view = out.parent / "pool_view"
    write_view(pool_view, out, "pilot_v1.1", small, {"train": s / "v1-all_tokens_train.txt"})
    return {**planned, "small": small, "files": files, "pool_view": str(pool_view)}


def test_view_checks_pass_on_consistent_views_and_catch_changes(views):
    v = load()
    out = views["out_dir"]
    assert not failed(v.check_view(out, views["pool_view"])), failed(v.check_view(out, views["pool_view"]))
    assert not failed(v.check_p12_view(out)) and not failed(v.check_views_train_only(out))
    # pool lines differ: the shared samples are not v1.0-pool's
    write_view(Path(views["pool_view"]), None, "pilot_v1.1", views["small"], {"train": views["s"] / "v1-all_tokens_train.txt"},
               tweak=lambda item: item.update(target="(9.0, 9.0)"))
    assert any("v1.0-pool's v1-all view" in n for n in failed(v.check_view(out, views["pool_view"])))
    # a view image that is not its original resized again
    folder = views["out"] / "views" / "pilot_v1" / "v1-full"
    write_view(folder, None, "pilot_v1.1", b"not the resized image", views["files"])
    assert any("resized again" in n for n in failed(v.check_view(out, None)))
    # the pilot_v1.2 view: another target, then other images than the pilot_v1.1 view's
    p12 = views["out"] / "views" / "pilot_v1.2" / "v1-full"
    write_view(p12, None, "pilot_v1.2", b"not the resized image", views["files"], prompt="Past poses",
               tweak=lambda item: item.update(target="(2.0, 0.0)"))
    names = failed(v.check_p12_view(out))
    assert any("line for line the pilot_v1.1 view's samples" in n for n in names)
    write_view(p12, None, "pilot_v1.2", views["small"], views["files"], prompt="Past poses")
    assert any("images.zip = the pilot_v1.1 view's" in n for n in failed(v.check_p12_view(out)))


def test_no_held_out_row_in_any_training_view(views):
    v = load()
    s = views["s"]
    held_row = next(t for t in tokens(s / "v1-full-devlogs_tokens_dev.txt") if t not in tokens(s / "v1-full_tokens_dev.txt"))
    leaked = views["out"] / "leaked_train.txt"
    scene_exclusion.write_tokens(leaked, tokens(s / "v1-full_tokens_train.txt") + [held_row])
    write_view(views["out"] / "views" / "pilot_v1.2" / "v1-full", None, "pilot_v1.2", views["small"],
               {"train": leaked, "dev": s / "v1-full_tokens_dev.txt"}, prompt="Past poses")
    assert any("no held-out row in any training view" in n for n in failed(v.check_views_train_only(views["out_dir"])))
    (views["out"] / "views" / "pilot_v1" / "v1-full-devlogs").mkdir(parents=True)
    assert any("no view of v1-full-devlogs" in n for n in failed(v.check_views_train_only(views["out_dir"])))


def test_links_point_at_both_views(views, tmp_path):
    v = load()
    links = tmp_path / "pilot_v1"
    links.mkdir()
    try:
        os.symlink(v.view_dir(views["out_dir"], "pilot_v1.1"), links / "v1-full", target_is_directory=True)
        os.symlink(v.view_dir(views["out_dir"], "pilot_v1.2"), links / "v1-full-p12", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links need extra rights here")
    assert not failed(v.check_links(views["out_dir"], str(links)))
    os.remove(links / "v1-full-p12")
    os.symlink(v.view_dir(views["out_dir"], "pilot_v1.1"), links / "v1-full-p12", target_is_directory=True)
    assert failed(v.check_links(views["out_dir"], str(links))) == [f"{links / 'v1-full-p12'} → the v1-full view (pilot_v1.2)"]
