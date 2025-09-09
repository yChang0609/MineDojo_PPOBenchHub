# Env
from env_factory import build_env

from tqdm import tqdm

import argparse
import pprint
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs/config.yaml')
parser.add_argument(
    '--mode', type=str,
    help='name of config file to load',
    default='random')
args = parser.parse_args()


def random_action(params):
    env = build_env(params)
    for i in tqdm(range(2), desc="Episode"):
        obs = env.reset()
        done = False
        pbar = tqdm(desc="Step")
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            pbar.update(1)
        print(f"{i+1}-th episode ran successful!")
    env.close()

def manual(params):
    from pynput import keyboard
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

    env = build_env(params,1)
    env.reset()
    total_reward = 0
    count_step = 0
    while True:
        action = [0,0]
        with keyboard.Events() as events:
            # Block for as much as possible
            event = events.get(1e6)
            if event.key == keyboard.KeyCode.from_char('q'):
                break
            # player move
            if event.key == keyboard.KeyCode.from_char('w'):
                action[0] = 1
            if event.key == keyboard.KeyCode.from_char('s'):
                action[0] = 2
            if event.key == keyboard.KeyCode.from_char('a'):
                action[0] = 3
            if event.key == keyboard.KeyCode.from_char('d'):
                action[0] = 4
            # view move
            if event.key == keyboard.KeyCode.from_char('i'):
                action[0] = 8
            if event.key == keyboard.KeyCode.from_char('k'):
                action[0] = 9
            if event.key == keyboard.KeyCode.from_char('j'):
                action[0] = 10
            if event.key == keyboard.KeyCode.from_char('l'):
                action[0] = 11
            # action
            if event.key == keyboard.KeyCode.from_char('v'):
                action[1] = 1
            if event.key == keyboard.KeyCode.from_char('b'):
                action[1] = 2
        obs, reward, done, info = env.step(action)
        count_step += 1
        total_reward += reward
        if count_step % 10 or reward > 0:
            print("total rewards:",total_reward)
        if done:
            count_step = 0
            total_reward = 0
            env.reset()

    env.close()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    if args.mode == "random":
        random_action(params)
    elif args.mode == "manual":
        manual(params)