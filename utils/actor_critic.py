import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Type, Optional, Dict, Any, Callable, Tuple

from .modules import *
# --------------------------------------------------------------------- #
#                      Policy / Value Networks                          
# --------------------------------------------------------------------- #

class DreamerNetwork(nn.Module):
    """
    Configurable backbone for policy and value networks.
    Supports:
      - hidden dim
      - number of layers
      - activation
      - normalization
      - shared or separate backbones
    """
    def __init__(
        self,
        feature_dim: int,
        hid_dim: int = 1024,
        num_layers: int = 3,
        activation: str = "silu",
        norm: Optional[str] = "rmsnorm",
        share_backbone: bool = False,
    ):
        super().__init__()
        self.latent_dim_pi = hid_dim
        self.latent_dim_vf = hid_dim
        self.share_backbone = share_backbone

        if share_backbone:
            self.shared = mlp(feature_dim, hid_dim, num_layers, activation, norm)
            self.policy_head = nn.Identity()
            self.value_head = nn.Identity()
        else:
            self.policy_net = mlp(feature_dim, hid_dim, num_layers, activation, norm)
            self.value_net = mlp(feature_dim, hid_dim, num_layers, activation, norm)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return latent policy and value representations."""
        if self.share_backbone:
            h = self.shared(features)
            return self.policy_head(h), self.value_head(h)
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.shared(features) if self.share_backbone else self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.shared(features) if self.share_backbone else self.value_net(features)


class DreamerActorCritic(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for SB3.
    Accepts `net_kwargs` in policy_kwargs for backbone configuration.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        self._net_kwargs: Dict[str, Any] = kwargs.pop("net_kwargs", {})
        kwargs["ortho_init"] = kwargs.get("ortho_init", False)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            normalize_images=False,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DreamerNetwork(
            feature_dim=self.features_dim,
            hid_dim=int(self._net_kwargs.get("hid_dim", 1024)),
            num_layers=int(self._net_kwargs.get("num_layers", 3)),
            activation=str(self._net_kwargs.get("activation", "silu")),
            norm=self._net_kwargs.get("norm", "rmsnorm"),
            share_backbone=bool(self._net_kwargs.get("share_backbone", False)),
        )
