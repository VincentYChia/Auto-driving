import hashlib
import io
import json
import os
import tarfile

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")
Image = pytest.importorskip("PIL.Image")

from data_pipeline.v1 import audit, navsim_splits, openscene_images, tables  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402


def jpeg(seed):
    buf = io.BytesIO()
    Image.new("RGB", (32, 18), (seed % 256, 80, 160)).save(buf, format="JPEG")
    return buf.getvalue()


def camera_archive(path, members):
    """The layout of openscene_sensor_trainval_camera_N.tgz: openscene-v1.1/sensor_blobs/trainval/<log>/<cam>/<file>."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(f"openscene-v1.1/sensor_blobs/trainval/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def indexed(tmp_path):
    logs = [v1.make_log("log_a", 60), v1.make_log("log_b", 60), v1.make_log("log_c", 60)]
    root = v1.write_navsim_root(tmp_path / "navsim", train=["log_a", "log_b"], val=["log_c"])
    out = tmp_path / "out"
    oi.build_index(str(v1.write_logs_dir(tmp_path / "logs", logs)), navsim_splits.load(str(root)), str(out))
    audit.main(["--out", str(out)])
    return out


def write_archives(where, out, broken_token=None, dropped_token=None):
    """Archive 0 holds log_a; archive 1 holds log_b and log_c. Every CAM_F0 frame plus a CAM_B0 one."""
    cands = tables.read(str(out / "index" / "candidates.parquet"), ["token", "log_name", "cam_f0_path"])
    src = where / "archives"
    src.mkdir(parents=True)
    layout = {0: ["log_a"], 1: ["log_b", "log_c"]}
    tree = []
    for n, logs in layout.items():
        members = []
        for k, (token, log, path) in enumerate(zip(cands["token"], cands["log_name"], cands["cam_f0_path"])):
            if log in logs and token != dropped_token:
                members.append((path, b"not a jpeg" if token == broken_token else jpeg(k)))
        members.append((f"{logs[0]}/CAM_B0/extra.jpg", jpeg(0)))
        name = openscene_images.ARCHIVE_NAME.format(n=n)
        data = camera_archive(src / name, members).read_bytes()
        tree.append({"type": "file", "path": f"openscene-v1.1/openscene_sensor_trainval_camera/{name}",
                     "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}})
    archive_map = where / "map.json"
    archive_map.write_text(json.dumps({f"openscene_sensor_trainval_{n}": logs for n, logs in layout.items()}))
    archive_tree = where / "tree.json"
    archive_tree.write_text(json.dumps(tree))
    return src, str(archive_map), str(archive_tree)


def read_tokens(out, name):
    return {t for split in ("train", "dev")
            for t in (out / "splits" / f"{name}_tokens_{split}.txt").read_text().split()}


def test_plan_selects_every_set_and_orders_archives(tmp_path, indexed):
    _, archive_map, archive_tree = write_archives(tmp_path, indexed)
    manifest = openscene_images.plan(str(indexed), archive_map, archive_tree)
    assert sorted(manifest["archive_order"]) == [0, 1] and manifest["index_logs_missing_from_map"] == 0
    pilot, smoke = read_tokens(indexed, "v1-pilot"), read_tokens(indexed, "smoke")
    assert pilot and smoke and manifest["frames_to_keep"] == len(pilot | smoke)
    assert sum(manifest["frames_to_keep_per_archive"].values()) == len(pilot | smoke)
    dev = (indexed / "splits" / "v1-pilot_tokens_dev.txt").read_text().split()
    assert dev and all(t.startswith("log_c") for t in dev)  # log_c is NAVSIM's val log
    assert manifest["selections"]["v1-pilot"]["config"]["balance_alpha"] > 0
    archives = json.loads((indexed / "images" / "archives.json").read_text())
    assert archives["1"]["logs"] == ["log_b", "log_c"] and archives["0"]["url"].endswith("camera_0.tgz")


def test_plan_for_v1_large_selects_only_it_and_keeps_the_pilot_dev_set(tmp_path, indexed):
    _, archive_map, archive_tree = write_archives(tmp_path, indexed)
    openscene_images.plan(str(indexed), archive_map, archive_tree)
    pilot_dev = (indexed / "splits" / "v1-pilot_tokens_dev.txt").read_text()
    manifest = openscene_images.plan(str(indexed), archive_map, archive_tree, set_name="v1-large")
    assert list(manifest["selections"]) == ["v1-large"] and manifest["smoke_archives"] == []
    assert (indexed / "splits" / "v1-large_tokens_dev.txt").read_text() == pilot_dev
    assert manifest["frames_to_keep"] == len(read_tokens(indexed, "v1-large"))


def test_fetch_from_a_source_dir_skips_archives_not_there_yet(tmp_path, indexed):
    src, archive_map, archive_tree = write_archives(tmp_path, indexed)
    openscene_images.plan(str(indexed), archive_map, archive_tree)
    later = src.parent / "later"
    later.mkdir()
    name = openscene_images.ARCHIVE_NAME.format(n=1)
    (src / name).rename(later / name)
    first = openscene_images.fetch(str(indexed), source_dir=str(src))
    assert first["done_now"] == [0] and first["not_in_source_dir"] == [1] and first["failed"] == []
    (later / name).rename(src / name)
    second = openscene_images.fetch(str(indexed), source_dir=str(src))
    assert second["done_now"] == [1] and second["not_in_source_dir"] == []


def test_fetch_shards_split_the_archives_between_processes(tmp_path, indexed):
    src, archive_map, archive_tree = write_archives(tmp_path, indexed)
    manifest = openscene_images.plan(str(indexed), archive_map, archive_tree)
    first = openscene_images.fetch(str(indexed), source_dir=str(src), shard=1, num_shards=2)
    assert first["done_now"] == [manifest["archive_order"][1]]
    second = openscene_images.fetch(str(indexed), source_dir=str(src), shard=0, num_shards=2)
    assert second["done_now"] == [manifest["archive_order"][0]] and second["archives_done"] == 2


def test_fetch_keeps_selected_jpegs_records_failures_and_resumes(tmp_path, indexed):
    cands = tables.read(str(indexed / "index" / "candidates.parquet"), ["token", "log_name"])
    openscene_images.plan(str(indexed), *write_archives(tmp_path / "clean", indexed)[1:])
    wanted = sorted(read_tokens(indexed, "v1-pilot") | read_tokens(indexed, "smoke"))
    broken, dropped = [t for t in wanted if t.startswith("log_b")][:2]
    # The selection depends on the metadata only, so planning again with damaged archives keeps it.
    src, archive_map, archive_tree = write_archives(tmp_path / "bad", indexed, broken_token=broken, dropped_token=dropped)
    openscene_images.plan(str(indexed), archive_map, archive_tree)
    assert sorted(read_tokens(indexed, "v1-pilot") | read_tokens(indexed, "smoke")) == wanted

    result = openscene_images.fetch(str(indexed), source_dir=str(src))
    assert result["failed"] == [] and sorted(result["done_now"]) == [0, 1]
    assert result["images_kept"] == len(wanted) - 2 and result["undecodable"] == 1 and result["missing"] == 1
    kept = {p.stem for p in (indexed / "images" / "CAM_F0").rglob("*.jpg")}
    assert kept == set(wanted) - {broken, dropped}
    log_of = dict(zip(cands["token"], cands["log_name"]))
    for token in kept:
        assert (indexed / "images" / "CAM_F0" / log_of[token] / f"{token}.jpg").exists()
    record = json.loads((indexed / "images" / "archives" / "1.json").read_text())
    assert record["logs_match_map"] and record["missing"] == [dropped] and record["undecodable"][0]["token"] == broken
    image = record["images"][0]
    path = indexed / "images" / "CAM_F0" / image["log"] / f"{image['token']}.jpg"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == image["sha256"] and [image["width"], image["height"]] == [32, 18]
    assert not list((indexed / "images" / "CAM_F0").rglob("*.part"))

    again = openscene_images.fetch(str(indexed), source_dir=str(src))
    assert again["done_now"] == [] and again["images_kept"] == result["images_kept"]  # finished archives are skipped


def test_fetch_with_parallel_downloads_extracts_in_order_and_cleans_up(tmp_path, indexed, monkeypatch):
    src, archive_map, archive_tree = write_archives(tmp_path, indexed)
    openscene_images.plan(str(indexed), archive_map, archive_tree)

    def download(archive, dest):  # stands in for the network: copy the local archive
        with open(src / archive["name"], "rb") as f, open(dest, "wb") as g:
            g.write(f.read())
        return 0.0

    monkeypatch.setattr(openscene_images, "_download", download)
    result = openscene_images.fetch(str(indexed), workers=2)
    order = json.loads((indexed / "splits" / "manifest.json").read_text())["archive_order"]
    assert result["failed"] == [] and result["done_now"] == order  # extracted in plan order
    assert result["images_kept"] == len(read_tokens(indexed, "v1-pilot") | read_tokens(indexed, "smoke"))
    assert not list((indexed / "images" / "tmp").iterdir())  # every downloaded archive deleted after extraction


def test_parallel_downloads_extract_in_plan_order_and_keep_the_same_images(tmp_path, indexed, monkeypatch):
    import shutil

    src, archive_map, archive_tree = write_archives(tmp_path, indexed)
    openscene_images.plan(str(indexed), archive_map, archive_tree)
    order = json.loads((indexed / "splits" / "manifest.json").read_text())["archive_order"]

    def download(archive, dest):  # stands in for the network: copy the local archive
        shutil.copyfile(src / archive["name"], dest)
        return 0.0

    monkeypatch.setattr(openscene_images, "_download", download)
    result = openscene_images.fetch(str(indexed), n_archives=len(order), workers=2)
    assert result["failed"] == [] and result["done_now"] == order  # extracted in plan order
    assert result["images_kept"] == len(read_tokens(indexed, "v1-pilot") | read_tokens(indexed, "smoke"))
    assert not list((indexed / "images" / "tmp").iterdir())  # every downloaded archive was deleted


def test_fetch_refuses_an_archive_whose_checksum_differs(tmp_path, indexed):
    src, archive_map, archive_tree = write_archives(tmp_path, indexed)
    openscene_images.plan(str(indexed), archive_map, archive_tree)
    with open(src / openscene_images.ARCHIVE_NAME.format(n=0), "ab") as f:
        f.write(b"tampered")
    result = openscene_images.fetch(str(indexed), source_dir=str(src))
    assert [f["archive"] for f in result["failed"]] == [0] and result["done_now"] == [1]
    assert not (indexed / "images" / "archives" / "0.json").exists()


def test_download_drops_a_slow_connection_and_resumes_where_it_stopped(tmp_path, monkeypatch, capsys):
    data = os.urandom(50_000)
    archive = {"name": "a.tgz", "url": "https://example.invalid/a.tgz", "bytes": len(data),
               "sha256": hashlib.sha256(data).hexdigest()}

    class Clock:  # fake time: the first connection delivers 1 kB per second, later ones are instant
        now = 0.0

        def time(self):
            return self.now

        def sleep(self, s):
            self.now += s

    clock, ranges = Clock(), []

    class Response:
        def __init__(self, start, seconds_per_read):
            self.pos, self.slow = start, seconds_per_read
            self.status = 206 if start else 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            clock.now += self.slow
            block = data[self.pos : self.pos + n]
            self.pos += len(block)
            return block

    def urlopen(request, timeout):
        start = int(request.get_header("Range", "bytes=0-")[6:].rstrip("-"))
        ranges.append(start)
        return Response(start, 1.0 if len(ranges) == 1 else 0.0)

    monkeypatch.setattr(openscene_images, "time", clock)
    monkeypatch.setattr(openscene_images, "CHUNK", 1000)
    monkeypatch.setattr(openscene_images, "SLOW_WINDOW_S", 10.0)
    monkeypatch.setattr(openscene_images.urllib.request, "urlopen", urlopen)
    dest = tmp_path / "a.tgz"
    openscene_images._download(archive, str(dest))
    assert dest.read_bytes() == data
    assert len(ranges) == 2 and ranges[0] == 0 and 0 < ranges[1] < len(data)  # resumed, not restarted
    assert "slow connection" in capsys.readouterr().out


def test_fetch_needs_full_for_more_than_the_smoke_archives(tmp_path, indexed):
    with pytest.raises(SystemExit):
        openscene_images.main(["fetch", "--out", str(indexed), "--archives", str(openscene_images.SMOKE_ARCHIVES + 1)])
