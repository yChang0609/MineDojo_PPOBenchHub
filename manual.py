# Env
import time
from src.utils import build_env, load_config

import argparse
from pynput import keyboard

import cv2
from PIL import Image
import numpy as np
import os

action_map = {
    1: [1, 0, 0],  # forward
    2: [2, 0, 0],  # backward
    3: [0, 1, 0],  # left
    4: [0, 2, 0],  # right
    5: [0, 0, 1],  # jump
    6: [0, 0, 2],  # sneak
    7: [0, 0, 3],  # sprint
    8: [0, 0, 0, 11, 12],  # camera pitch +30
    9: [0, 0, 0, 13, 12],  # camera pitch -30
    10: [0, 0, 0, 12, 11],  # camera yaw +30
    11: [0, 0, 0, 12, 13]  # camera yaw -30
}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs/config.yaml')
args = parser.parse_args()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    params = load_config(args.config)
    task = params["Environment"]["task"]
    model_name = params["PPO_Training"]["save_name"]

    env = build_env(params,1)

    env.reset()
    print("Play!!")
    total_reward = 0
    count_step = 0
    while True:
        action = [0,0]
        with keyboard.Events() as events:
            # Block for as much as possible
            event = events.get(1e6)
            if event.key == keyboard.KeyCode.from_char('q'):
                break
            if event.key == keyboard.KeyCode.from_char('w'):
                action[0] = 1
            if event.key == keyboard.KeyCode.from_char('s'):
                action[0] = 2
            if event.key == keyboard.KeyCode.from_char('a'):
                action[0] = 3
            if event.key == keyboard.KeyCode.from_char('d'):
                action[0] = 4

            if event.key == keyboard.KeyCode.from_char('i'):
                action[0] = 8
            if event.key == keyboard.KeyCode.from_char('k'):
                action[0] = 9
            if event.key == keyboard.KeyCode.from_char('j'):
                action[0] = 10
            if event.key == keyboard.KeyCode.from_char('l'):
                action[0] = 11
            
            if event.key == keyboard.KeyCode.from_char('v'):
                action[1] = 1
            if event.key == keyboard.KeyCode.from_char('b'):
                action[1] = 2
        obs, reward, done, info = env.step(action)
        count_step += 1
        total_reward += reward
        if count_step % 10:
            print("total rewards:",total_reward)
        if done:
            count_step = 0
            total_reward = 0
            env.reset()

    env.close()