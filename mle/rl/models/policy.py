from math import log
from loguru import logger
from mle.utils import model_utils
import torch as th

import abc
from torch import distributions as td, nn


class BasePolicy(abc.ABC, nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abc.abstractmethod
    def act(self, state) -> th.Tensor: ...

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

    def act(self, state) -> th.Tensor:
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
        use_full_network_for_log_std: bool = False,
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.mu = model_utils.build_simple_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
        )
        self.use_full_network_for_log_std = use_full_network_for_log_std
        if use_full_network_for_log_std:
            self.log_std = model_utils.build_simple_mlp(
                input_dim=state_dim,
                output_dim=action_dim,
                n_hidden_layers=n_hidden_layers,
                hidden_dim=hidden_dim,
            )
        else:
            self.log_std = nn.Parameter(th.zeros((action_dim,)))

    def cov_var(self, state) -> th.Tensor:
        eps = 1e-5
        if self.use_full_network_for_log_std:
            log_std = self.log_std(state)
        else:
            log_std = self.log_std
        return th.diag(log_std.exp().square() + eps)

    def action_dist(self, state) -> td.MultivariateNormal:
        return td.MultivariateNormal(
            loc=self.mu(state),
            covariance_matrix=self.cov_var(state),
        )


class TanhNormalPolicy(BasePolicy):
    """
    A stochastic policy that is multivariate gaussian with a tanh call at the end so outputs are bounded
    between [-1, 1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_hidden_layers: int,
        hidden_dim: int,
    ):
        super().__init__(state_dim, action_dim)
        self.mu = model_utils.build_simple_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
        )
        self.log_std = model_utils.build_simple_mlp(
            input_dim=state_dim,
            output_dim=action_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_dim=hidden_dim,
        )

    def act(self, state) -> th.Tensor:
        std = self.log_std(state).exp()
        mu = self.mu(state)
        noise = th.randn_like(mu)
        action = th.tanh(mu + (noise * std))
        return action

    def log_prob(self, state, action) -> th.Tensor:
        mu = self.mu(state)
        covar = th.diag_embed(self.log_std(state).exp().square() + 1e-6)
        action = th.clip(action, -1 + 1e-6, 1 - 1e-6)
        pre_tanh_action = th.atanh(action)
        log_prob = th.distributions.MultivariateNormal(mu, covar).log_prob(
            pre_tanh_action
        ) - (1 - action.pow(2)).log().sum(dim=1)  # determinant of jacobian of tanh(x)

        return log_prob
