# PPO

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from ppo_model import build_ppo

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


def ppo_training(params):
    
    task = params["Environment"]["task"]
    num_envs = params["Environment"]["num_envs"]
    seed = params["Environment"]["seed"]

    model_name = params["PPO_Training"]["save_name"]
    total_timesteps = params["PPO_Training"]["training_step"]

    vec_env = make_vec_env(
        lambda : build_env(params, seed),
        n_envs=num_envs
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    import os
    import shutil
    mount_path_env = os.getenv('MOUNT_PATH', "")
    log_dir = os.path.join(mount_path_env, f"logs/ppo_{task}/{model_name}")
    


    model, episode_logger_callback = build_ppo(
        **params["PPO_Training"]["policy_network"],
        vec_env=vec_env, num_envs=num_envs, 
        log_dir=log_dir
    )
    print(model.policy.pi_features_extractor)
    print(model.policy.mlp_extractor)

    model.learn(total_timesteps=total_timesteps, callback=episode_logger_callback)
    
    dummy_config_path = os.path.join(model.logger.dir, f"config.yaml")
    model_save_dir = os.path.join(model.logger.dir, model_name)

    shutil.copy(args.config, dummy_config_path)
    model.save(model_save_dir)
    vec_env.close()
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
        ppo_training(params)
