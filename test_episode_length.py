"""Debug script to trace why episode_length is only 1."""

import os
import numpy as np

os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/home/yutianqi/.egl/10_nvidia.json"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HABITAT_DATA_PATH"] = "/home/yutianqi/data"
os.environ["HABITAT_SIM_LOG"] = "quiet"

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

print("=== Testing episode termination ===")
for episode in range(5):
    print(f"\n--- Episode {episode+1} ---")
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs, info = result, {}
    
    print(f"Reset info: {info}")
    
    step_count = 0
    done = False
    total_reward = 0.0
    
    while not done and step_count < 50:
        action = np.zeros(11, dtype=np.float32)  # 11-dim action
        result = env.step(action)
        
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done_old, info = result
            terminated, truncated = done_old, False
        
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: reward={reward:.6f}, terminated={terminated}, truncated={truncated}, "
              f"force_terminate={info.get('force_terminate', 'N/A')}, does_want_terminate={info.get('does_want_terminate', 'N/A')}")
        
        if terminated or truncated:
            done = True
            print(f"Episode ended at step {step_count}!")
            print(f"Final reward: {total_reward:.6f}")
            print(f"Final info: {info}")

env.close()