from __future__ import annotations
from typing import Type, Optional, Dict, Any, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ----------------------------- #
#       Episode Logger           #
# ----------------------------- #
class EpisodeLoggerCallback(BaseCallback):
    """Log episode rewards/lengths per env."""
    def __init__(self, num_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.num_envs = num_envs
        self.episode_steps = [0] * num_envs
        self.episode_rewards = [0.0] * num_envs
        self.episode_count = [0] * num_envs

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        for i in range(self.num_envs):
            self.episode_rewards[i] += float(rewards[i])
            self.episode_steps[i] += 1
            if bool(dones[i]):
                self.episode_count[i] += 1
                self.logger.record(f"Episode/Env{i}/Steps", self.episode_steps[i])
                self.logger.record(f"Episode/Env{i}/Reward", self.episode_rewards[i])
                self.logger.dump(self.num_timesteps)
                self.episode_steps[i] = 0
                self.episode_rewards[i] = 0.0
        return True


# ----------------------------- #
#         PPO Builder            #
# ----------------------------- #
def build_ppo(
    vec_env,
    num_envs: int,
    *,
    policy: Optional[Type[ActorCriticPolicy]] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    ent_coef: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    log_dir: Optional[str] = None,
    verbose: int = 1,
    **ppo_kwargs: Any,
) -> Tuple[PPO, BaseCallback]:
    """
    Assemble a PPO agent using a user-provided policy class + kwargs.
    Adds validation for features extractor settings.
    """
    if policy is None:
        policy = ActorCriticPolicy
    if policy_kwargs is None:
        policy_kwargs = {}

    # --- Validation for features extractor ---
    if "features_extractor_class" in policy_kwargs:
        fe_cls = policy_kwargs["features_extractor_class"]
        if not issubclass(fe_cls, BaseFeaturesExtractor):
            raise TypeError(
                f"features_extractor_class must be a subclass of BaseFeaturesExtractor, got {fe_cls}"
            )
        if "features_extractor_kwargs" not in policy_kwargs:
            raise ValueError(
                "policy_kwargs must include 'features_extractor_kwargs' "
                "when 'features_extractor_class' is provided."
            )
        if "features_dim" not in policy_kwargs["features_extractor_kwargs"]:
            raise ValueError(
                "features_extractor_kwargs must include 'features_dim' "
                "(the output feature dimension expected by the policy)."
            )

    if not issubclass(policy, ActorCriticPolicy):
        raise TypeError("policy must be a subclass of ActorCriticPolicy.")

    ep_cb = EpisodeLoggerCallback(num_envs=num_envs, verbose=1)

    algo = PPO(
        policy=policy,
        env=vec_env,
        policy_kwargs=policy_kwargs,
        ent_coef=float(ent_coef),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        tensorboard_log=log_dir,
        verbose=verbose,
        **ppo_kwargs,
    )
    return algo, ep_cb
