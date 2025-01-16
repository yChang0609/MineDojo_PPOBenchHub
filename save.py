# PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from ppo_model import build_ppo

# Env
import time
from src.utils import build_env, load_config

import argparse
from utils import load_config

import cv2
from PIL import Image
import numpy as np
import os



parser = argparse.ArgumentParser()
parser.add_argument(
    "-log", 
    required=True
)
parser.add_argument(
    "-export", 
    required=True
)
args = parser.parse_args()
mount_path_env = os.getenv('MOUNT_PATH', "")

def save_images(image_list, folder_path, prefix="step"):
    os.makedirs(folder_path, exist_ok=True)
    
    for i, img_array in enumerate(image_list):
        img = Image.fromarray(np.uint8(img_array))
        
        file_path = os.path.join(folder_path, f"{prefix}_{i + 1}.png")
        img.save(file_path)

def vec_env_obs2obs_list(vec_env_obs,n_stack=1):
    obs = vec_env_obs.squeeze(0)
    split_obs = np.split(obs, n_stack, axis=0)
    return [(_obs.transpose(1, 2, 0)) for _obs in split_obs]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    params = load_config(os.path.join(args.log, "config.yaml"))
    save_images_path = os.path.join(mount_path_env,f"{args.export}")
    os.makedirs(save_images_path, exist_ok=True)

    task = params["Environment"]["task"]
    model_name = params["PPO_Training"]["save_name"]
    total_success_avg = 0

    model = PPO.load(os.path.join(f"{args.log}",model_name))

    seed = 456
    n_stack = 4
    vec_env = make_vec_env(lambda: build_env(params,seed), n_envs=1)
    vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    obs = vec_env.reset()
    success_count = 0
    save_frames = []

    total_steps = 0
    total_reward = 0
    episode_reward=[]
    while True:
        action, _ = model.predict(obs.copy())
        obs, reward, done, info = vec_env.step(action)
        # if (len(episode_reward) + 1) % 10 == 0:
        save_frames += vec_env_obs2obs_list(obs, n_stack=n_stack)
        total_steps += 1
        if done:
            break

    vec_env.close()
    
    print(f"total_steps: {total_steps}")
    print(f"saving images...")
    save_images(save_frames, args.export,"step")

    print(f"success save.")




