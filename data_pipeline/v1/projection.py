"""Project points in the ego frame onto the native (distorted) CAM_F0 image.

Conventions, assumed here and checked by eye in the viewer (step 4 checks them properly):
* ``sensor2lidar_rotation`` / ``_translation`` map camera coordinates to the LiDAR frame:
  ``p_lidar = R @ p_cam + t`` (OpenScene's naming). The camera frame is OpenCV's: x right, y down,
  z forward.
* ``lidar2ego`` maps the LiDAR frame to the ego frame (rear axle, x forward, y left). In the
  OpenScene trainval metadata it is the identity for every log (Gate 0 audit).
* The ego frame's origin is on the ground, so a path pose ``(x, y)`` is the point ``(x, y, 0)``.
  The camera then sits ~1.5 m up (audit: ``cam_translation_m.z``). Flat ground is assumed: the
  stored path has no height.
* ``distortion`` is OpenCV's ``(k1, k2, p1, p2, k3)``; the images are native, not rectified.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

MIN_DEPTH_M = 1.0  # drop points closer to the camera plane than this
MAX_RADIUS = 1.2  # normalised image radius beyond which the distortion polynomial is not trusted


def quaternion_matrix(wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    n = w * w + x * x + y * y + z * z
    if n == 0:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def ego_to_camera(points: np.ndarray, sensor2lidar_rotation, sensor2lidar_translation,
                  lidar2ego_rotation=(1.0, 0.0, 0.0, 0.0), lidar2ego_translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    """``(N, 3)`` ego-frame points in the camera frame."""
    r_cam = np.asarray(sensor2lidar_rotation, dtype=np.float64).reshape(3, 3)
    t_cam = np.asarray(sensor2lidar_translation, dtype=np.float64).reshape(3)
    r_lid = quaternion_matrix(lidar2ego_rotation)
    t_lid = np.asarray(lidar2ego_translation, dtype=np.float64).reshape(3)
    lidar = (np.asarray(points, dtype=np.float64) - t_lid) @ r_lid  # row-vector form of R^T (p - t)
    return (lidar - t_cam) @ r_cam


def distort(xn: np.ndarray, yn: np.ndarray, k) -> tuple:
    """OpenCV's forward distortion of normalised coordinates."""
    k1, k2, p1, p2, k3 = (float(v) for v in k)
    r2 = xn * xn + yn * yn
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return xd, yd


def project(points_cam: np.ndarray, intrinsic, distortion, size: Optional[tuple] = None) -> np.ndarray:
    """``(N, 2)`` pixel coordinates; NaN where a point is behind the camera, too wide for the
    distortion model, or (with ``size = (width, height)``) outside the image."""
    p = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    out = np.full((len(p), 2), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
    ok = (p[:, 2] > MIN_DEPTH_M) & (xn * xn + yn * yn < MAX_RADIUS ** 2)
    xd, yd = distort(xn[ok], yn[ok], distortion)
    out[ok, 0] = k[0, 0] * xd + k[0, 1] * yd + k[0, 2]
    out[ok, 1] = k[1, 1] * yd + k[1, 2]
    if size is not None:
        w, h = size
        outside = (out[:, 0] < 0) | (out[:, 0] > w) | (out[:, 1] < 0) | (out[:, 1] > h)
        out[outside] = np.nan
    return out


def path_pixels(path_xyyaw: np.ndarray, valid: np.ndarray, log: dict, size: Optional[tuple] = None) -> np.ndarray:
    """Pixels of a stored path (ego frame at the anchor, on flat ground) for one log's calibration.

    ``log`` holds the ``logs.parquet`` calibration columns of that log.
    """
    path = np.asarray(path_xyyaw, dtype=np.float64).reshape(-1, 3)
    ground = np.column_stack([path[:, 0], path[:, 1], np.zeros(len(path))])
    cam = ego_to_camera(ground, log["cam_f0_sensor2lidar_rotation"], log["cam_f0_sensor2lidar_translation"],
                        log["lidar2ego_rotation"], log["lidar2ego_translation"])
    px = project(cam, log["cam_f0_intrinsic"], log["cam_f0_distortion"], size)
    px[~np.asarray(valid, dtype=bool)] = np.nan
    return px
