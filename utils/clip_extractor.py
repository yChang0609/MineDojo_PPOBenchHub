import torch
import hashlib
import torch.nn as nn
from gym import spaces
from omegaconf import OmegaConf
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from mineclip import MineCLIP
except ImportError:
    MineCLIP = None

from .modules import *

def load_mineclip(clip_model_path: str, verify_checksum: bool = True):
    """
    Load MineCLIP from a given path and freeze parameters.
    Raises ImportError if mineclip is not installed.
    """
    if MineCLIP is None:
        raise ImportError(
            "MineCLIP is not available. Please install the `mineclip` package "
            "or avoid using features_extractor_type='CLIP'."
        )

    cfg = OmegaConf.load(f"{clip_model_path}/conf.yaml")
    OmegaConf.set_struct(cfg, False)
    ckpt = cfg.pop("ckpt")
    OmegaConf.set_struct(cfg, True)
    if verify_checksum:
        md5 = hashlib.md5(open(ckpt.path, "rb").read()).hexdigest()
        assert md5 == ckpt.checksum, f"Checkpoint checksum mismatch: {md5} vs {ckpt.checksum}"
    model = MineCLIP(**cfg)
    model.load_ckpt(ckpt.path, strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


class CLIPFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using MineCLIP.
    Expects input shape [B, C, H, W] where C = stack_frames * 3.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        stack_frames: int,
        clip_model_path: str,
        features_dim: int = 256,
        project_activation: str = "relu",
    ):
        super().__init__(observation_space, features_dim)
        self.stack_frames = int(stack_frames)
        self.requirement_channels = 3

        self.clip = load_mineclip(clip_model_path)
        clip_out_dim = int(self.clip.clip_model.vision_model.output_dim)

        self.proj = nn.Sequential(
            nn.Linear(clip_out_dim, features_dim),
            get_activation(project_activation),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: [B, C, H, W]
        Output: [B, features_dim]
        """
        B, C, H, W = observations.shape
        assert C == self.stack_frames * self.requirement_channels, \
            f"Expected {self.stack_frames*3} channels, got {C}"

        device = observations.device
        self.clip.to(device)

        video = observations.view(B, self.stack_frames, self.requirement_channels, H, W).contiguous()
        with torch.inference_mode():
            x = self.clip.encode_video(video)  # [B, D]
        return self.proj(x)
