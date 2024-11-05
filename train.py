from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import src.tasks as tasks 
import src.core.actions as actions
import src.core.observation as observation
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

def ppo_training(params, env):
    env = make_vec_env(lambda: env, n_envs=4)
    env = VecFrameStack(env, n_stack=4)

    task = params["Environment"]["task"]
    name = params["PPO_Training"]["save_name"]
    total_timesteps = params["PPO_Training"]["training_step"]

    log_dir = f".logs/ppo_{task}/{name}"
    model = PPO("MlpPolicy", env, ent_coef=0.01, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps)
    model.save("./model/"+ name)
    env.close()

def random_action(params, env):
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
    params = None
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    env = tasks.available_task[params["Environment"]["task"]]( # "CombatSpider"
        **params["Environment"]["task_parameter"]
        # image_size=(224,224)
        # step_penalty=0,
        # attack_reward=1,
        # success_reward=10,
    )
    env = actions.available_action_space[params["Environment"]["action_space"]]( #"ReducedActionSpace"
        env=env
    )
    env = observation.available_observation[params["Environment"]["observation"]]( #"ImageObservation"
        env=env
    )
    mode = params["Mode"]
    if "PPO_Training" == mode:
        ppo_training(params, env)
    elif "Random" == mode:
        random_action(params, env)