import io
import json
import tarfile

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("yaml")

from data_pipeline.v1 import audit, navsim_splits, projection, viewer  # noqa: E402
from data_pipeline.v1 import openscene_index as oi  # noqa: E402
from tests.data import v1_synthetic as v1  # noqa: E402

# A forward-looking camera 1.5 m ahead of the rear axle and 1.5 m up: camera (x right, y down,
# z forward) to ego (x forward, y left, z up).
CAM_TO_EGO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
CAM_POS = np.array([1.5, 0.0, 1.5])
K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])


def calib(distortion=np.zeros(5)):
    return {
        "cam_f0_intrinsic": K.reshape(-1), "cam_f0_distortion": distortion,
        "cam_f0_sensor2lidar_rotation": CAM_TO_EGO.reshape(-1), "cam_f0_sensor2lidar_translation": CAM_POS,
        "lidar2ego_rotation": np.array([1.0, 0.0, 0.0, 0.0]), "lidar2ego_translation": np.zeros(3),
    }


def test_projection_matches_a_pinhole_camera():
    path = np.array([[11.5, 0.0, 0.0], [11.5, 2.0, 0.0], [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    px = projection.path_pixels(path, np.ones(4, bool), calib())
    assert px[0] == pytest.approx([960.0, 540.0 + 1000.0 * 1.5 / 10.0])  # 10 m ahead of the camera, 1.5 m below it
    assert px[1][0] == pytest.approx(960.0 - 1000.0 * 2.0 / 10.0)  # left in the ego frame is left in the image
    assert np.isnan(px[2]).all() and np.isnan(px[3]).all()  # behind the camera / closer than MIN_DEPTH_M
    hidden = projection.path_pixels(path, np.array([False, True, True, True]), calib())
    assert np.isnan(hidden[0]).all()
    off_image = projection.path_pixels(np.array([[2.6, 0.0, 0.0]]), np.ones(1, bool), calib(), size=(1920, 1080))
    assert np.isnan(off_image).all()  # 1.1 m ahead and 1.5 m down falls below the image


def test_barrel_distortion_pulls_points_towards_the_centre():
    path = np.array([[11.5, 3.0, 0.0]])
    plain = projection.path_pixels(path, np.ones(1, bool), calib())[0]
    barrel = projection.path_pixels(path, np.ones(1, bool), calib(np.array([-0.3, 0.0, 0.0, 0.0, 0.0])))[0]
    assert abs(barrel[0] - 960) < abs(plain[0] - 960) and abs(barrel[1] - 540) < abs(plain[1] - 540)


def test_lidar2ego_rotation_is_applied():
    yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # LiDAR x axis = ego y axis
    np.testing.assert_allclose(projection.quaternion_matrix(yaw90) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    point = np.array([[0.0, 5.0, 0.0]])
    rotated = projection.ego_to_camera(point, np.eye(3), np.zeros(3), yaw90, np.zeros(3))
    np.testing.assert_allclose(rotated, [[5.0, 0.0, 0.0]], atol=1e-12)


def camera_archive(path, members):
    """The layout of openscene_sensor_trainval_camera_N.tgz: openscene-v1.1/sensor_blobs/trainval/<log>/<cam>/<file>."""
    with tarfile.open(path, "w:gz") as tar:
        for name in members:
            data = b"\xff\xd8 jpeg " + name.encode()
            info = tarfile.TarInfo(f"openscene-v1.1/sensor_blobs/trainval/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_fetch_keeps_indexed_front_images_and_build_writes_the_page(tmp_path):
    logs = [v1.make_log("log_a", 40), v1.make_log("log_b", 40)]
    root = v1.write_navsim_root(tmp_path / "navsim", train=["log_a"], val=["log_b"])
    out = tmp_path / "out"
    oi.build_index(str(v1.write_logs_dir(tmp_path / "logs", logs)), navsim_splits.load(str(root)), str(out))
    audit.main(["--out", str(out)])
    members = ["log_a/CAM_F0/0005.jpg", "log_a/CAM_F0/0006.jpg", "log_a/CAM_B0/0005.jpg",
               "log_a/CAM_F0/0000.jpg",  # too close to the log's start: not a candidate
               "log_c/CAM_F0/0005.jpg"]  # not in the index
    result = viewer.fetch(str(out), source=str(camera_archive(tmp_path / "cam.tgz", members)))
    assert result["images_kept"] == 2 and result["files_seen"] == 5
    kept = sorted(p.relative_to(out / "viewer" / "sensor_blobs").as_posix() for p in (out / "viewer" / "sensor_blobs").rglob("*.jpg"))
    assert kept == ["log_a/CAM_F0/0005.jpg", "log_a/CAM_F0/0006.jpg"]

    page = viewer.build(str(out), sample=5)
    assert (out / "viewer" / "index.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    text = (out / "viewer" / "data.js").read_text(encoding="utf-8")
    assert text.startswith("window.VIEWER_DATA = ") and page.endswith("index.html")
    data = json.loads(text[len("window.VIEWER_DATA = "):].rstrip().rstrip(";"))
    rows = data["rows"]
    with_image = [r for r in rows if r["img"]]
    assert [r["img"] for r in with_image] == kept and all("px" in r for r in with_image)
    assert data["meta"]["image_logs"] == ["log_a"] and data["meta"]["rows_with_image"] == 2
    others = [r for r in rows if not r["img"]]
    assert len(others) == 5 and {r["log"] for r in others} == {"log_b"}  # the sample skips logs with images
    assert all(len(r["path"]) == len(oi.PATH_OFFSETS_S) and len(r["why"]) == 3 for r in rows)
    assert data["summary"]["logs"]["total"] == 2
