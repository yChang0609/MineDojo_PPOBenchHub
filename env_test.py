# Env
from src.utils import build_env

from tqdm import tqdm

import argparse
import pprint
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs/config.yaml')
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

if __name__ == "__main__":
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
        random_action(params)