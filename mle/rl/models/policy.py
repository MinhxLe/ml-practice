from mle.utils import model_utils

import abc
from torch import distributions as td, nn


class BasePolicy(abc.ABC, nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def action_dist(self, state) -> td.Distribution: ...

    def act(self, state):
        return self.action_dist(state).sample()

    def forward(self, state):
        return self.action_dist(state).sample()


class DiscretePolicy(BasePolicy):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        n_actions: int,
    ):
        super().__init__(state_dim=state_dim, action_dim=n_actions)
        self.layers = model_utils.build_simple_mlp(
            input_dim=state_dim,
            output_dim=n_actions,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
        )

    def action_dist(self, state) -> td.Categorical:
        return td.Categorical(logits=self.layers(state))
