# PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from utils.ppo_model import build_ppo

# Env
import time
from env_factory import build_env

import argparse
import pprint
import yaml

import cv2
import numpy as np
import os



parser = argparse.ArgumentParser()
parser.add_argument(
    "-logs", 
    nargs='+', 
    help="Log file names",
    required=True
)
args = parser.parse_args()
mount_path_env = os.getenv('MOUNT_PATH', "")

def vec_env_obs2obs_list(vec_env_obs,n_stack=1):
    obs = vec_env_obs.squeeze(0)
    split_obs = np.split(obs, n_stack, axis=0)
    return [(cv2.cvtColor(_obs.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)) for _obs in split_obs]

def load_config(config_path):
    params = None
    with open(config_path, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)
    return params

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    for log in args.logs:
        params = load_config(os.path.join(log, "config.yaml"))
        save_name = params["PPO_Training"]["save_name"]
        eval_result_csv_path = os.path.join(mount_path_env,f"eval_result/{save_name}/")
        eval_result_video_path = os.path.join(mount_path_env,f"eval_result/{save_name}/")
        os.makedirs(eval_result_csv_path, exist_ok=True)
        os.makedirs(eval_result_video_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = 30
        frame_size = tuple(reversed(params["Environment"]["task_parameter"]["image_size"]))

        task = params["Environment"]["task"]
        model_name = params["PPO_Training"]["save_name"]
        eval_episode = 20 # params["PPO_Training"]["eval_episode"]
        total_success_avg = 0

        seed=123
        n_stack=4
        dummy_vec_env = make_vec_env(
            lambda : build_env(params, seed),
            n_envs=1
        )
        dummy_vec_env = VecFrameStack(dummy_vec_env, n_stack=n_stack)
        log_dir = f"logs/ppo_{task}/{model_name}"
        
        model, episode_logger_callback = build_ppo(
            **params["PPO_Training"]["policy_network"],
            vec_env=dummy_vec_env, num_envs=1, 
            log_dir=log,
            stack_frame=n_stack
        )
        model.set_parameters(f"{log}/" + model_name)
        dummy_vec_env.close()

        eval_seed_list = [456, 789, 357, 468, 790]
        n_stack = 4
        for seed in eval_seed_list:
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
                total_reward += reward
                total_steps += 1

                if done:
                    # episode_reward append and reset 
                    episode_reward.append(total_reward)
                    print(f"Episode-{len(episode_reward)} / step: {total_steps} & reward: {total_reward}")
                    
                    # insert done_frame
                    # if (len(episode_reward)) % 10 == 0:
                    done_frame = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
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
                    if total_steps < params["Environment"]["task_parameter"]["max_episode_len"]:
                            success_count += 1
                    if len(episode_reward) == eval_episode:        
                        out = cv2.VideoWriter(f"{eval_result_video_path}/episodes{eval_episode}_{sum(episode_reward)/len(episode_reward)}.mp4", fourcc, fps, frame_size)
                        for frame in save_frames:
                            out.write(frame)
                        out.release()
                        break
                    # reset env
                    total_steps = 0
                    total_reward = 0
                    obs = vec_env.reset()
            vec_env.close()
            total_success_avg += (success_count / eval_episode * 100)  
            print(f"success_rate: {(success_count / eval_episode * 100):.2f}%")
            print(f"Avg reward for ep{eval_episode}: {sum(episode_reward)/len(episode_reward)}")
        total_success_avg /= 5
        print(f"total success rate: {(total_success_avg):.2f}%")

