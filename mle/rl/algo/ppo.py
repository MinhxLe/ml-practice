from typing import Callable
import torch
from torch import nn
from mle.rl.core import calculate_returns

from mle.rl.algo.policy_gradient import (
    PolicyGradient,
    PolicyGradientCfg,
    PolicyTrainer,
)
from mle.rl.core import Trajectory
from mle.rl.env import GymEnv
from mle.rl.models.policy import BasePolicy
import attrs


@attrs.frozen(kw_only=True)
class PPOCfg(PolicyGradientCfg):
    clipped_eps: float
    n_gradient_steps: int


class PPOPolicyTrainer(PolicyTrainer):
    def __init__(
        self,
        policy: BasePolicy,
        baseline: nn.Module | None,
        lr: float,
        gamma: float,
        eps: float,
        n_steps: int,
    ):
        super().__init__(policy, baseline, lr=lr, gamma=gamma)
        assert 0 < eps < 1
        self.eps = eps
        self.n_steps = n_steps

    def loss_fn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        original_action_log_probs: torch.Tensor,
    ):
        action_log_probs = self.model.action_dist(states).log_prob(actions)
        ratio = torch.exp(action_log_probs - original_action_log_probs)
        clipped_ratio = torch.clip(ratio, 1 - self.eps, 1 + self.eps)
        return -torch.minimum(advantages * ratio, advantages * clipped_ratio).mean()

    def update(self, trajs: list[Trajectory]) -> float:
        traj_tds = [t.to_tensordict() for t in trajs]
        all_states = torch.concat([t["state"] for t in traj_tds])
        all_actions = torch.concat([t["action"] for t in traj_tds])
        all_advantages = []
        all_returns = []
        for traj, traj_td in zip(trajs, traj_tds):
            returns = calculate_returns(traj, self.gamma)
            advantages = self._calculate_advantages(returns, traj_td["state"])
            all_advantages.append(advantages)
            all_returns.append(returns)
        all_advantages = torch.concat(all_advantages)
        all_returns = torch.concat(all_returns)
        with torch.no_grad():
            original_action_log_probs = self.model.action_dist(all_states).log_prob(
                all_actions
            )

        total_loss = 0.0
        for _ in range(self.n_steps):
            total_loss += self._step_optimizer(
                states=all_states,
                advantages=all_advantages,
                actions=all_actions,
                original_action_log_probs=original_action_log_probs,
            )
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

    def _init_policy_trainer(self):
        return PPOPolicyTrainer(
            policy=self.policy,
            baseline=self.baseline,
            lr=self.cfg.lr,
            gamma=self.cfg.gamma,
            eps=self.cfg.clipped_eps,
            n_steps=self.cfg.n_gradient_steps,
        )
