# PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# Env
from src.utils import build_env

import argparse
import pprint
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs/config.yaml')
args = parser.parse_args()

if __name__ == "__main__":
    params = None
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    vec_env = make_vec_env(lambda: build_env(params), n_envs=1)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model_name = params["PPO_Training"]["save_name"]
    eval_episode = params["PPO_Training"]["eval_episode"]
    model = PPO.load("model/"+model_name)

    obs = vec_env.reset()
    total_reward = 0
    episode_reward=[]
    while True:
        action, _ = model.predict(obs.copy())
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward
        if done:
            episode_reward.append(total_reward)
            print(f"Episode-{len(episode_reward)} reward: {total_reward}")
            total_reward = 0
            obs = vec_env.reset()
            if len(episode_reward) == eval_episode:        
                break
    print(f"Avg reward for ep{eval_episode}: {sum(episode_reward)/len(episode_reward)}")