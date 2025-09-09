# convert_all.py
import yaml
import argparse
import os

def convert(old_cfg: dict) -> dict:
    """Convert old config dict to new config format."""
    env_cfg = old_cfg.get("Environment", {})
    train_cfg = old_cfg.get("PPO_Training", {})

    new_cfg = {
        "Environment": {
            "task": env_cfg.get("task"),
            "task_parameter": env_cfg.get("task_parameter", {}),
            "num_envs": env_cfg.get("num_envs", 1),
            "seed": env_cfg.get("seed", 0),
            "action_space": env_cfg.get("action_space"),
            "observation": env_cfg.get("observation"),
            "frame_stack": 4,  # default
        },
        "PPO_Training": {
            "save_name": train_cfg.get("save_name", "ppo_run"),
            "total_timesteps": train_cfg.get("training_step", 1_000_000),
            "eval_episodes": train_cfg.get("eval_episode", 10),
            "algo": {
                "gamma": train_cfg.get("policy_network", {}).get("gamma", 0.99),
                "gae_lambda": train_cfg.get("policy_network", {}).get("gea_lambda", 0.95),  # typo fix
                "ent_coef": train_cfg.get("policy_network", {}).get("entropy_coef", 0.0),
            },
            "policy": {
                "class_path": "utils.policies.DreamerActorCritic",
                "ortho_init": False,
                "net_kwargs": {
                    "hid_dim": 1024,
                    "num_layers": 3,
                    "activation": "silu",
                    "norm": "rmsnorm",
                    "share_backbone": False,
                },
            },
        },
    }

    feat_type = train_cfg.get("policy_network", {}).get("features_extractor_type", "CNN")
    if feat_type.upper() == "CLIP":
        new_cfg["PPO_Training"]["policy"]["features_extractor"] = {
            "class_path": "utils.extractors.CLIPFeatureExtractor",
            "kwargs": {
                "features_dim": 1024,
                "clip_model_path": train_cfg.get("policy_network", {}).get("mine_clip_path", "./mineclip_model"),
                "project_activation": "relu",
            },
        }
    else:
        new_cfg["PPO_Training"]["policy"]["features_extractor"] = {
            "class_path": "utils.extractors.CNNFeatureExtractor",
            "kwargs": {
                "features_dim": 1024,
                "project_activation": "relu",
            },
        }

    return new_cfg


def process_folder(folder: str, output_folder: str):
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.endswith((".yaml", ".yml")):
                old_path = os.path.join(root, fname)

                rel_path = os.path.relpath(old_path, folder)
                new_path = os.path.join(output_folder, rel_path)

                os.makedirs(os.path.dirname(new_path), exist_ok=True)

                with open(old_path, "r") as f:
                    old_cfg = yaml.load(f, Loader=yaml.FullLoader)

                try:
                    new_cfg = convert(old_cfg)
                except Exception as e:
                    print(f"[!] Failed to convert {old_path}: {e}")
                    continue

                with open(new_path, "w") as f:
                    yaml.dump(new_cfg, f, sort_keys=False)

                print(f"Converted {old_path} -> {new_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="folder path with old YAML configs")
    parser.add_argument("--output", type=str, default="converted", help="output folder path")
    args = parser.parse_args()

    process_folder(args.input, args.output)
