import torch
from torch import nn
from mle.rl.core import calculate_returns

from mle.rl.algo.policy_gradient import (
    BaselineTrainer,
    PolicyGradient,
    PolicyGradientCfg,
    PolicyTrainer,
)
from mle.rl.core import Trajectory
from mle.rl.models.policy import BasePolicy
import attrs


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
        action_log_probs = self.policy.action_dist(states).log_prob(actions)
        ratio = torch.exp(action_log_probs - original_action_log_probs)
        clipped_ratio = torch.clip(ratio, min=1 - self.eps, max=1 + self.eps)
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
            original_action_log_probs = (
                self.policy.action_dist(all_states).log_prob(all_actions).detach()
            )

        total_loss = 0.0
        for _ in range(self.n_steps):
            self.optimizer.zero_grad()
            loss = self.loss_fn(
                states=all_states,
                advantages=all_advantages,
                actions=all_actions,
                original_action_log_probs=original_action_log_probs,
            )
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / self.n_steps


@attrs.frozen(kw_only=True)
class PPOCfg(PolicyGradientCfg):
    clipped_eps: float
    n_gradient_steps: int


class PPO(PolicyGradient):
    def _init_policy_trainer(self):
        return PPOPolicyTrainer(
            policy=self.policy,
            baseline=self.baseline,
            lr=self.cfg.lr,
            gamma=self.cfg.gamma,
            eps=self.cfg.clipped_eps,
            n_steps=self.cfg.n_gradient_steps,
        )

    def _init_baseline_trainer(self) -> BaselineTrainer | None:
        cfg = self.cfg
        if self.baseline:
            baseline_trainer = BaselineTrainer(
                self.baseline,
                lr=cfg.lr,
                gamma=cfg.gamma,
                n_steps=self.cfg.n_gradient_steps,
            )
        else:
            baseline_trainer = None
        return baseline_trainer
