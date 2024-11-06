# PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

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

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, num_envs, verbose=0):
        super(EpisodeLoggerCallback, self).__init__(verbose)
        self.num_envs = num_envs
        self.episode_steps = [0] * num_envs   
        self.episode_rewards = [0] * num_envs
    
    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']

        for env_idx in range(self.num_envs):
            self.episode_rewards[env_idx] += rewards[env_idx]
            self.episode_steps[env_idx] += 1

            if dones[env_idx]:
                self.logger.record(f"Episode/Env_{env_idx}/Episode_Steps", self.episode_steps[env_idx])
                self.logger.record(f"Episode/Env_{env_idx}/Episode_Reward", self.episode_rewards[env_idx])
                self.logger.dump(self.num_timesteps)

                self.episode_steps[env_idx] = 0
                self.episode_rewards[env_idx] = 0

        return True
    
def ppo_training(params):
    
    task = params["Environment"]["task"]
    model_name = params["PPO_Training"]["save_name"]
    total_timesteps = params["PPO_Training"]["training_step"]
    policy = params["PPO_Training"]["policy"]
    num_envs = params["Environment"]["num_envs"]
    seed = params["Environment"]["seed"]


    vec_env = make_vec_env(
        lambda : build_env(params, seed),
        n_envs=num_envs
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    log_dir = f"logs/ppo_{task}/{model_name}"
    episode_logger_callback = EpisodeLoggerCallback(num_envs=num_envs, verbose=1)
    model = PPO(policy, vec_env, ent_coef=0.01, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps, callback=episode_logger_callback)
    model.save("./model/"+ model_name)
    vec_env.close()
    
if __name__ == "__main__":
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
        ppo_training(params)
