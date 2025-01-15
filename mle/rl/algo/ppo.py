from typing import Callable
import torch
from torch import nn, optim

from mle.rl.algo.policy_gradient import (
    PolicyGradient,
    PolicyGradientCfg,
)
from mle.rl.env import GymEnv
from mle.rl.models.policy import BasePolicy
from mle.trainer import BaseTrainer
import attrs


@attrs.frozen(kw_only=True)
class PPOCfg(PolicyGradientCfg):
    clipped_eps: float
    n_gradient_steps: int


class PPOPolicyTrainer(BaseTrainer):
    def __init__(self, policy: BasePolicy, lr: float, eps: float, n_steps: int):
        super().__init__(policy, optim.Adam, dict(lr=lr))
        assert 0 < eps < 1
        self.eps = eps
        self.n_steps = n_steps

    def loss_fn(
        self,
        advantages: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        old_action_log_probs: torch.Tensor,
    ):
        action_log_probs = self.model.action_dist(states).log_prob(actions)
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        clipped_ratio = torch.clip(ratio, 1 - self.eps, 1 + self.eps)
        return -torch.minimum(advantages * ratio, advantages * clipped_ratio).mean()

    def update(self, **kwargs) -> float:
        total_loss = 0.0
        for _ in range(self.n_steps):
            total_loss += super().update(**kwargs)
        return total_loss / self.n_steps


class PPO(PolicyGradient):
    def __init__(
        self,
        create_policy_fn: Callable[[], BasePolicy],
        create_baseline_fn: Callable[[], nn.Module] | None,
        env: GymEnv,
        cfg: PPOCfg,
    ):
        super().__init__(create_policy_fn, create_baseline_fn, env, cfg)

    def _init_policy_trainer(self, policy, cfg):
        return PPOPolicyTrainer(policy, cfg.lr, cfg.clipped_eps, cfg.n_gradient_steps)
