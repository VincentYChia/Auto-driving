"""Synthetic NAVSIM raw frame dicts with analytically known motion, for tests.

Ego: constant speed V on a constant-yaw-rate arc. Agent "car": straight line at constant
global velocity. Agent "ped": appears from frame 2 on, velocity NaN. Agent "far": 300 m away.
Field names follow NAVSIM's log format (see navsim_tools/raw_frames.py).
"""

import math

import numpy as np

DT = 0.5
V = 10.0          # ego speed, m/s
W = 0.08          # ego yaw rate, rad/s
YAW0 = 0.4        # ego yaw at t = 0 (frame 0), rad
P0 = np.array([1000.0, 500.0])
CAR_P0 = np.array([1030.0, 515.0])
CAR_V = np.array([6.0, -2.0])     # global velocity of the car
PED_P = np.array([1015.0, 512.0])  # static pedestrian position (global)
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")  # OpenScene rig


def ego_global(t: float) -> np.ndarray:
    yaw = YAW0 + W * t
    x = P0[0] + (V / W) * (math.sin(yaw) - math.sin(YAW0))
    y = P0[1] - (V / W) * (math.cos(yaw) - math.cos(YAW0))
    return np.array([x, y, yaw])


def car_global(t: float) -> np.ndarray:
    pos = CAR_P0 + CAR_V * t
    return np.array([pos[0], pos[1], math.atan2(CAR_V[1], CAR_V[0])])


def to_local(pose_global: np.ndarray, ego: np.ndarray) -> np.ndarray:
    c, s = math.cos(-ego[2]), math.sin(-ego[2])
    dx, dy = pose_global[0] - ego[0], pose_global[1] - ego[1]
    yaw = math.atan2(math.sin(pose_global[2] - ego[2]), math.cos(pose_global[2] - ego[2]))
    return np.array([c * dx - s * dy, s * dx + c * dy, yaw])


def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def make_frames(num_frames: int = 14):
    frames = []
    for i in range(num_frames):
        t = i * DT
        ego = ego_global(t)
        boxes, names, vels, tokens = [], [], [], []
        car_local = to_local(car_global(t), ego)
        boxes.append([car_local[0], car_local[1], 0.8, 4.5, 1.9, 1.6, car_local[2]])
        names.append("vehicle")
        vels.append(list(rotate(CAR_V, -ego[2])) + [0.0])
        tokens.append("track_car")
        if i >= 2:
            ped_local = to_local(np.array([PED_P[0], PED_P[1], 1.0]), ego)
            boxes.append([ped_local[0], ped_local[1], 0.9, 0.6, 0.6, 1.7, ped_local[2]])
            names.append("pedestrian")
            vels.append([float("nan"), float("nan"), 0.0])
            tokens.append("track_ped")
        boxes.append([300.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0])
        names.append("vehicle")
        vels.append([0.0, 0.0, 0.0])
        tokens.append("track_far")
        frames.append(
            {
                "token": f"tok{i:02d}",
                "timestamp": 1_000_000 + int(t * 1e6),
                "log_name": "synthetic_log",
                "scene_token": "scene0",
                "map_location": "us-nv-las-vegas-strip",
                "roadblock_ids": ["rb1"],
                "traffic_lights": [(123, False)],
                "ego2global_translation": [ego[0], ego[1], 0.0],
                "ego2global_rotation": [math.cos(ego[2] / 2), 0.0, 0.0, math.sin(ego[2] / 2)],
                "ego_dynamic_state": [V, 0.0, 0.0, V * W],
                "driving_command": [0, 1, 0, 0],
                "anns": {
                    "gt_boxes": np.array(boxes, dtype=np.float32),
                    "gt_names": names,
                    "gt_velocity_3d": np.array(vels, dtype=np.float32),
                    "instance_tokens": [f"inst_{tok}_{i}" for tok in tokens],
                    "track_tokens": tokens,
                },
                "cams": {
                    name: {
                        "data_path": f"synthetic_log/{name}/{i:04d}.jpg",
                        "cam_intrinsic": np.eye(3),
                        "sensor2lidar_rotation": np.eye(3),
                        "sensor2lidar_translation": np.zeros(3),
                        "distortion": np.zeros(5),
                    }
                    for name in CAMERAS
                },
                "lidar_path": f"synthetic_log/MergedPointCloud/{i:04d}.pcd",
            }
        )
    return frames
