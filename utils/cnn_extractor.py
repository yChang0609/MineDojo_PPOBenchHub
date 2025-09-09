import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .modules import *

class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Simple CNN feature extractor with customizable conv layers.
    conv_spec: List of (out_channels, kernel_size, stride, padding).
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        conv_spec: Optional[List[Tuple[int, int, int, int]]] = None,
        project_activation: str = "relu",
        input_scale: float = 255.0,
    ):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        conv_spec = conv_spec or [
            (32, 8, 4, 0),
            (64, 4, 2, 0),
        ]
        conv_layers: List[nn.Module] = []
        in_ch = n_input_channels
        for out_ch, k, s, p in conv_spec:
            conv_layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p), nn.ReLU()]
            in_ch = out_ch
        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)

        # Compute output dimension
        with torch.no_grad():
            dummy = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            get_activation(project_activation),
        )
        self.input_scale = float(input_scale)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations / self.input_scale
        return self.linear(self.cnn(x))