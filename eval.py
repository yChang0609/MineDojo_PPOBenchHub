# PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

# Env
from src.utils import build_env

import argparse
import pprint
import yaml

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str,
    help='name of config file to load',
    default='configs/config.yaml')
args = parser.parse_args()

def vec_env_obs2obs_list(vec_env_obs):
    obs = vec_env_obs.squeeze(0)
    split_obs = np.split(obs, 4, axis=0)
    return [(cv2.cvtColor(_obs.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)) for _obs in split_obs]

if __name__ == "__main__":
    params = None
    with open(args.config, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    vec_env = make_vec_env(lambda: build_env(params), n_envs=1)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    task = params["Environment"]["task"]
    model_name = params["PPO_Training"]["save_name"]
    eval_episode = params["PPO_Training"]["eval_episode"]
    
    model = PPO.load("model/"+model_name)
    log_dir = f"logs/ppo_{task}/{model_name}"

    save_frames = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    fps = 30
    frame_size = (224, 224) 

    obs = vec_env.reset()
    save_frames += vec_env_obs2obs_list(obs)
    
    total_reward = 0
    episode_reward=[]
    while True:
        action, _ = model.predict(obs.copy())
        obs, reward, done, info = vec_env.step(action)
        save_frames += vec_env_obs2obs_list(obs)
        total_reward += reward
        
        if done:
            # episode_reward append and reset 
            episode_reward.append(total_reward)
            print(f"Episode-{len(episode_reward)} reward: {total_reward}")
            
            # insert done_frame
            done_frame = np.ones((224, 224, 3), dtype=np.uint8) * 255
            text = f"Ep{len(episode_reward)}:{total_reward}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (done_frame.shape[1] - text_width) // 2  
            text_y = (done_frame.shape[0] + text_height) // 2  
            position = (text_x, text_y)  
            cv2.putText(
                done_frame, 
                text, position, cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2,
                cv2.LINE_AA      
            )
            save_frames += [done_frame]*16

            total_reward = 0
            obs = vec_env.reset()
            if len(episode_reward) == eval_episode:        
                out = cv2.VideoWriter(f"{log_dir}/episodes{eval_episode}_{sum(episode_reward)/len(episode_reward)}.mp4", fourcc, fps, frame_size)
                for frame in save_frames:
                    out.write(frame)
                out.release()
                break
    print(f"Avg reward for ep{eval_episode}: {sum(episode_reward)/len(episode_reward)}")