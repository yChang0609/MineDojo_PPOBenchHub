import torch
import torch.nn as nn
from typing import Optional, List

def get_activation(name: str) -> nn.Module:
    """Return activation layer by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ["silu", "swish"]:
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")

def get_norm(name: Optional[str], dim: int) -> nn.Module:
    """Return normalization layer by name."""
    if not name or name.lower() == "none":
        return nn.Identity()
    name = name.lower()
    if hasattr(nn, "RMSNorm") and name == "rmsnorm":
        return nn.RMSNorm(dim)  # torch >= 2.0
    if name in ["layernorm", "ln"]:
        return nn.LayerNorm(dim)
    if name in ["batchnorm1d", "bn1d"]:
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unknown norm: {name}")

def mlp(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "silu",
    norm: Optional[str] = "rmsnorm",
    out_dim: Optional[int] = None,
) -> nn.Sequential:
    """
    Build a configurable MLP with [Linear -> Norm -> Act] repeated `num_layers` times.
    Optionally add a final output projection layer.
    """
    layers: List[nn.Module] = []
    last_dim = in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(last_dim, hidden_dim), get_norm(norm, hidden_dim), get_activation(activation)]
        last_dim = hidden_dim
    if out_dim is not None:
        layers.append(nn.Linear(last_dim, out_dim))
    return nn.Sequential(*layers)
