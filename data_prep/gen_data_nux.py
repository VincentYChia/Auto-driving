import os
from dotenv import load_dotenv
import numpy as np
from math import atan2
import json

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

load_dotenv()

VERSION = 'v1.0-trainval'
DATAROOT = os.getenv("NUSCENES_ROOT")
DATA_DIR = os.getenv("VLAD_DATA_DIR", "data")
# output dir（ train / val）
PROMPT_TYPE = ["long", "short"][0]
OUTPUT_TEXT = True
if OUTPUT_TEXT:
    OUTPUT_JSON_TRAIN = os.path.join(DATA_DIR, f"nuscenes_waypoint_text_{PROMPT_TYPE}_prompt_train_Nu_X.json")
    OUTPUT_JSON_VAL   = os.path.join(DATA_DIR, f"nuscenes_waypoint_text_{PROMPT_TYPE}_prompt_val_Nu_X.json")
else:
    OUTPUT_JSON_TRAIN = os.path.join(DATAROOT, f"nuscenes_waypoint_{PROMPT_TYPE}_prompt_train.json")
    OUTPUT_JSON_VAL   = os.path.join(DATAROOT, f"nuscenes_waypoint_{PROMPT_TYPE}_prompt_val.json")

# Nu-X raw annotations (narration + reasoning per sample_token).
NU_X_TRAIN_JSON = os.path.join(DATA_DIR, "Nu_X_train.json")
NU_X_VAL_JSON = os.path.join(DATA_DIR, "Nu_X_val.json")


def load_nu_x_annotations(path):
    if not path or not os.path.exists(path):
        print(f"[warn] Nu_X annotation file not found: {path}")
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[info] Loaded {len(data)} Nu_X annotations from {path}")
        return data
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Failed to load Nu_X annotations from {path}: {exc}")
        return {}


NU_X_ANNOTATIONS = {
    'train': load_nu_x_annotations(NU_X_TRAIN_JSON),
    'val': load_nu_x_annotations(NU_X_VAL_JSON),
}

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
scenes = nusc.scene

splits = create_splits_scenes()
train_scene_names = set(splits['train'])
val_scene_names   = set(splits['val'])

def convert_with_format(num):
    return "{:012d}".format(num)

def yaw_from_quat(q):
    """
    q: [w, x, y, z]
    """
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return atan2(siny_cosp, cosy_cosp)

def world_to_ego0_xy(p_world_xy, p0_world_xy, yaw0):
    dx = p_world_xy[0] - p0_world_xy[0]
    dy = p_world_xy[1] - p0_world_xy[1]
    cy, sy = np.cos(-yaw0), np.sin(-yaw0)
    x_ego = cy * dx - sy * dy
    y_ego = sy * dx + cy * dy
    return float(x_ego), float(y_ego)

def format_waypoints_str(pts_xy, prec=2):
    def fmt(v): 
        return f"{v:.{prec}f}"
    return ", ".join([f"[{fmt(x)}, {fmt(y)}]" for x, y in pts_xy])

def history_waypoints_ego0(obs_traj_world, yaw0, p0_xy, prec=2):
    hist = []
    for p in obs_traj_world:
        x, y = world_to_ego0_xy((p[0], p[1]), p0_xy, yaw0)
        hist.append([x, y])
    return hist

# --------- （train / val） ----------
data_train = []
data_val = []
idx_train = 1
idx_val = 1

for scene in scenes:
    name = scene['name']
    target_split = None
    if name in train_scene_names:
        target_split = 'train'
    elif name in val_scene_names:
        target_split = 'val'
    else:
        continue

    token = scene['token']
    first_sample_token = scene['first_sample_token']
    last_sample_token  = scene['last_sample_token']
    description = scene['description']

    front_camera_images = []
    sample_tokens = []
    ego_poses = []
    camera_params = []
    curr_sample_token = first_sample_token

    while True:
        sample = nusc.get('sample', curr_sample_token)
        cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

        front_camera_images.append(
            os.path.join(nusc.dataroot, cam_front_data['filename'])
        )

        sample_tokens.append(curr_sample_token)
        pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        ego_poses.append(pose)

        camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample['next']

    scene_length = len(front_camera_images)
    print(f"[{target_split}] Scene {name} has {scene_length} frames")

    if scene_length < TTL_LEN:
        print(f"[{target_split}] Scene {name} has less than {TTL_LEN} frames, skipping...")
        continue

    ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

    for i in range(scene_length - TTL_LEN + 1):
        obs_images = front_camera_images[i:i+OBS_LEN]
        obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
        fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]

        t0_pose = ego_poses[i + OBS_LEN - 1]
        p0 = t0_pose['translation']  # [x, y, z]
        yaw0 = yaw_from_quat(t0_pose['rotation'])
        p0_xy = (float(p0[0]), float(p0[1]))

        hist_waypoints = history_waypoints_ego0(obs_ego_traj_world, yaw0, p0_xy, prec=2)
        hist_waypoints_str = format_waypoints_str(hist_waypoints)

        fut_waypoints_ego = []
        for p in fut_ego_traj_world:
            xk, yk = world_to_ego0_xy((p[0], p[1]), p0_xy, yaw0)
            fut_waypoints_ego.append([xk, yk])
        fut_waypoints_str = format_waypoints_str(fut_waypoints_ego)

        if PROMPT_TYPE == "short":
            system_message = (
                f"Predict the next {FUT_LEN} ego-frame waypoints (x-forward, y-left) at 0.5s intervals based on front-view image."
            )
            user_prompt = ""
        elif PROMPT_TYPE == "long":
            system_message = (
                "You are an autonomous driving labeller. You are given a sequence of front-view images, "
                "the 5-second historical ego waypoints in the *current ego frame* (x-forward, y-left), "
                "and a driving rationale. Your task is to predict the future ego-frame waypoints for the next "
                f"{FUT_LEN} timesteps (≈0.5 s interval). Provide raw coordinates only.\n"
            )

            sample_token = sample_tokens[i + OBS_LEN - 1]
            nu_x_entry = NU_X_ANNOTATIONS[target_split].get(sample_token, {})
            rationale_lines = []
            narration = nu_x_entry.get('narration')
            if narration:
                rationale_lines.append(f"Narration: {narration}")
            reasoning = nu_x_entry.get('reasoning')
            if reasoning:
                rationale_lines.append(f"Reasoning: {reasoning}")
            rationale_section = "\n".join(rationale_lines) + "\n" if rationale_lines else ""

            user_prompt = (
                "This is a frame from a front camera video captured.\n"
                f"The 5-second historical ego-frame waypoints (relative to the last observed frame) are {hist_waypoints_str}.\n"
                f"Generate the predicted ten future waypoints in the action token format, with the narration and reasoning:"
            )
            

        cur_data = {
            'id': convert_with_format(idx_train if target_split == 'train' else idx_val),
            'image': obs_images[-1],
            "action_targets": fut_waypoints_ego,
            "action_mask": [1] * len(fut_waypoints_ego),
            "scene": name,
            "conversations": [
                {
                    "from": "human",
                    "value": system_message + '<image>' + user_prompt
                },
                {
                    "from": "gpt",
                    "value": (fut_waypoints_str if OUTPUT_TEXT else "") + "\n" + rationale_section
                }
            ]
        }

        if target_split == 'train':
            data_train.append(cur_data)
            idx_train += 1
        else:
            data_val.append(cur_data)
            idx_val += 1

os.makedirs(os.path.dirname(OUTPUT_JSON_TRAIN), exist_ok=True)
with open(OUTPUT_JSON_TRAIN, 'w', encoding='utf-8') as f:
    json.dump(data_train, f, indent=2, ensure_ascii=False)
with open(OUTPUT_JSON_VAL, 'w', encoding='utf-8') as f:
    json.dump(data_val, f, indent=2, ensure_ascii=False)

print(f"[ok] wrote train samples: {len(data_train)} -> {OUTPUT_JSON_TRAIN}")
print(f"[ok] wrote val   samples: {len(data_val)}   -> {OUTPUT_JSON_VAL}")
