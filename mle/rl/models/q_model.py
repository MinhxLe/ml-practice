import abc
import torch
from torch import nn

from mle.utils import model_utils


class BaseQModel(abc.ABC, nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def forward(self, state, action):
        pass


class SimpleQModel(BaseQModel):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_hidden_layers: int,
        hidden_dim: int,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.layers = model_utils.build_simple_mlp(
            input_dim=state_dim + action_dim,
            output_dim=1,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
        )

    def forward(self, state, action):
        return self.layers(torch.concat([state, action], dim=1))
