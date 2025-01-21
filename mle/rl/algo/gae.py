"""
implementation of GAE estimator of advantage for VPG
"""

import attrs
import torch as th
from torch import nn
from mle.rl.algo.policy_gradient import BasePolicyTrainer
from mle.rl.algo.vpg import VPG, VPGCfg
from mle.rl.core import (
    Trajectory,
    Transitions,
    calculate_discounted_cumsum,
)
from loguru import logger
from mle.rl.models.policy import BasePolicy


class GAETrainer(BasePolicyTrainer):
    def __init__(
        self,
        policy: BasePolicy,
        baseline: nn.Module,
        lr: float,
        gamma: float,
        lambda_: float,
    ):
        super().__init__(policy, baseline, lr, gamma)
        self.lambda_ = lambda_

    def _calculate_advantages(self, transitions: Transitions) -> th.Tensor:
        terminated = transitions.terminated.to(th.int)
        values = self.baseline(transitions.state).squeeze(1)
        next_state_values = (terminated - 1) * self.baseline(
            transitions.next_state
        ).squeeze(1)
        tds = (transitions.reward + self.gamma * values) - next_state_values
        return calculate_discounted_cumsum(tds, self.gamma * self.lambda_)

    def loss_fn(
        self,
        states: th.Tensor,
        actions: th.Tensor,
        advantages: th.Tensor,
    ):
        action_log_probs = self.policy.action_dist(states).log_prob(actions)
        return -(action_log_probs * advantages).mean()

    def update(self, trajs: list[Trajectory]) -> float:
        transitions_list = [t.to_transitions() for t in trajs]
        all_states = th.concat([t.state for t in transitions_list])
        all_actions = th.concat([t.action for t in transitions_list])
        all_advantages = th.concat(
            [self._calculate_advantages(t) for t in transitions_list]
        )
        logger.debug(all_advantages.max().item())
        all_advantages = (all_advantages - all_advantages.mean()) / all_advantages.std()
        loss = self.loss_fn(
            states=all_states, advantages=all_advantages, actions=all_actions
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


@attrs.define
class VPGWithGAECfg(VPGCfg):
    lambda_: float


class VPGWithGAE(VPG):
    def _init_policy_trainer(self):
        return GAETrainer(
            self.policy,
            lr=self.cfg.lr,
            baseline=self.baseline,
            gamma=self.cfg.gamma,
            lambda_=self.cfg.lambda_,
        )
