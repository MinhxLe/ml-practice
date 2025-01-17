from mle.utils import model_utils
import torch

import abc
from torch import distributions as td, nn


class BasePolicy(abc.ABC, nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def act(self, state) -> torch.Tensor: ...

    def forward(self, state):
        return self.act(state)


class SimplePolicy(BasePolicy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_hidden_layers: int,
        hidden_dim: int,
    ):
        super().__init__(state_dim, action_dim)
        self.layers = model_utils.build_simple_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
        )

    def act(self, state) -> torch.Tensor:
        return self.layers(state)


class BaseStochasticPolicy(BasePolicy, abc.ABC):
    @abc.abstractmethod
    def action_dist(self, state) -> td.Distribution: ...

    def act(self, state):
        return self.action_dist(state).sample()


class CategoricalPolicy(BaseStochasticPolicy):
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


class GaussianPolicy(BaseStochasticPolicy):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        action_dim: int,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.mu = model_utils.build_simple_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
        )
        # [TODO] make it a function of the input
        self.log_std = nn.Parameter(torch.zeros((action_dim,)))

    def action_dist(self, state) -> td.MultivariateNormal:
        return td.MultivariateNormal(
            loc=self.mu(state),
            # add a small value to be PSD
            covariance_matrix=torch.diag(self.log_std.exp().square() + 1e-5),
        )
