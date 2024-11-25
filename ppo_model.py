from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable, Tuple
import torch
import torch.nn as nn
from gym import spaces


# -- MineCLIP
import hashlib
import hydra
from omegaconf import OmegaConf
from mineclip import MineCLIP


def load_clip(mount_path):
    cfg = OmegaConf.load(f"{mount_path}/conf.yaml")
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)
    assert (
        hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum
    ), "broken ckpt"
    model = MineCLIP(**cfg)
    model.load_ckpt(ckpt.path, strict=True)
    return model

class CLIPFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, clip_model_path, features_dim: int = 256, ):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        print(f"Load MineCLIP model from:{clip_model_path}")
        self.clip = load_clip(clip_model_path).to("cuda:0")
        for param in self.clip.parameters():
            param.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(self.clip.clip_model.vision_model.output_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        B, C, H, W = observations.shape
        requirement_chanels = 3
        stack_frame = C // requirement_chanels
        reshaped_obs = observations.view(B, stack_frame, requirement_chanels, H, W)
        with torch.no_grad():
            x = self.clip.forward_image_features(reshaped_obs)
        return self.linear(torch.mean(x, dim=1))

class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class DreamerNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        hid_dim:int = 1024
    ):
        super().__init__()
        # last_layer_dim = hid_dim
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = hid_dim #last_layer_dim_pi
        self.latent_dim_vf = hid_dim #last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hid_dim), nn.RMSNorm(hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.RMSNorm(hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.RMSNorm(hid_dim), nn.SiLU(),
            # nn.Linear(1024, last_layer_dim_pi)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hid_dim), nn.RMSNorm(hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.RMSNorm(hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.RMSNorm(hid_dim), nn.SiLU(),
            # nn.Linear(1024, last_layer_dim_vf)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)
    
class DreamerActorCritic(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DreamerNetwork(self.features_dim)

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, num_envs, verbose=0):
        super(EpisodeLoggerCallback, self).__init__(verbose)
        self.num_envs = num_envs
        self.episode_steps = [0] * num_envs   
        self.episode_rewards = [0] * num_envs
        self.episode_count = [0] * num_envs
    
    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']

        for env_idx in range(self.num_envs):
            self.episode_rewards[env_idx] += rewards[env_idx]
            self.episode_steps[env_idx] += 1

            if dones[env_idx]:
                self.episode_count[env_idx] += 1
                self.logger.record(f"Episode/Env_{env_idx}/Episode_Steps", self.episode_steps[env_idx])
                self.logger.record(f"Episode/Env_{env_idx}/Episode_Reward", self.episode_rewards[env_idx])
                self.logger.dump(self.episode_count[env_idx])
                # actions = self.locals.get('actions', None)[env_idx]
                # print(f"End action:{actions} / is_drop:{actions[5] == 2} , is_destroy:{actions[5] == 7}")
                # if not actions[5] == 2 and not actions[5] == 7:
                #      print(f"End action:{actions}")


                self.episode_steps[env_idx] = 0
                self.episode_rewards[env_idx] = 0

        return True
    
def build_ppo(vec_env, num_envs, 
              entropy_coef, gamma, gea_lambda,
              log_dir, features_extractor_type, **kw
              )-> tuple[PPO, BaseCallback] :
    episode_logger_callback = EpisodeLoggerCallback(num_envs=num_envs, verbose=1)
    
    if features_extractor_type == "CLIP":
        clip_path = kw.get('mine_clip_path', None)
        assert not clip_path == None
        policy_kwargs = dict(
            features_extractor_class=CLIPFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=1024,
                clip_model_path=clip_path
                ),

            )
        
    elif features_extractor_type == "CNN":
        policy_kwargs = dict(
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=1024),
            )
    else:
        policy_kwargs = None

    return PPO(
        DreamerActorCritic, vec_env,
        policy_kwargs=policy_kwargs,
        ent_coef=float(entropy_coef), gamma=gamma, gae_lambda=gea_lambda,
        verbose=1, tensorboard_log=log_dir), \
        episode_logger_callback
     
