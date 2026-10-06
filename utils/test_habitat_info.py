"""Test script to inspect Habitat environment's info dict structure."""

import os
import sys
import numpy as np

os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/home/yutianqi/.egl/10_nvidia.json"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HABITAT_DATA_PATH"] = "/home/yutianqi/data"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# Change to project root so Habitat can find data/
os.chdir("/home/yutianqi")

from habitat.gym.gym_definitions import get_config, make_gym_from_config

config_path = "benchmark/rearrange/multi_task/rearrange_easy.yaml"
data_path = os.environ.get("HABITAT_DATA_PATH", "/home/yutianqi/data")
overrides = [
    f"habitat.dataset.data_path={data_path}/datasets/replica_cad/rearrange/v2/train/rearrange_easy.json.gz",
    f"habitat.dataset.scenes_dir={data_path}/versioned_data/replica_cad_dataset",
    f"habitat.simulator.scene_dataset={data_path}/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json",
    "habitat.seed=42",
]

habitat_cfg = get_config(config_path, overrides=overrides)
env = make_gym_from_config(habitat_cfg)

print("=== Reset info ===")
result = env.reset()
if isinstance(result, tuple):
    obs, info = result
else:
    obs, info = result, {}
print(f"Info type: {type(info)}")
print(f"Info keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
if isinstance(info, dict):
    for k, v in info.items():
        print(f"  {k}: {type(v).__name__} = {v}")

print("\n=== Step info (10 steps) ===")
for i in range(10):
    action = np.zeros(11, dtype=np.float32)  # 11-dim action
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    else:
        obs, reward, done, info = result
        terminated, truncated = done, False
    
    if i == 0:
        print(f"Info keys: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
        if isinstance(info, dict):
            for k, v in info.items():
                print(f"  {k}: {type(v).__name__}")
    
    if terminated or truncated:
        print(f"\n=== Episode ended at step {i+1} ===")
        print(f"reward: {reward}")
        print(f"terminated: {terminated}, truncated: {truncated}")
        print(f"info: {info}")
        break

env.close()