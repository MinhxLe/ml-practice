"""
https://arxiv.org/pdf/1802.09477

OPEN QUESTION
"""

import attrs
import torch as th
from typing import Callable
from mle.rl.algo.off_policy import BaseOffPolicy, BaseOffPolicyCfg
from mle.rl.core import Transitions
from torch.nn import functional as F
from mle.rl.models.policy import BasePolicy
from mle.rl.models.q_model import BaseQModel
from mle.utils import model_utils, train_utils


@attrs.frozen
class TD3Cfg(BaseOffPolicyCfg):
    polyak_update_factor: float  # how much to update the target network

    act_noise_scale: float
    target_act_noise_scale: float
    target_act_noise_clip: float

    policy_lr: float
    q_model_lr: float


@attrs.define
class TD3(BaseOffPolicy[TD3Cfg]):
    create_q_model_fn: Callable[[], BaseQModel]
    q_model1: BaseQModel = attrs.field(init=False)
    target_q_model1: BaseQModel = attrs.field(init=False)
    q_model_optimizer1: th.optim.Adam = attrs.field(init=False)
    q_model2: BaseQModel = attrs.field(init=False)
    target_q_model2: BaseQModel = attrs.field(init=False)
    q_model_optimizer2: th.optim.Adam = attrs.field(init=False)
    target_policy: BasePolicy = attrs.field(init=False)
    policy_optimizer: th.optim.Adam = attrs.field(init=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.q_model1 = self.create_q_model_fn()
        self.q_model2 = self.create_q_model_fn()
        self.target_q_model1 = model_utils.copy_model(self.q_model1)
        self.target_q_model2 = model_utils.copy_model(self.q_model2)
        self.target_policy = model_utils.copy_model(self.policy)
        self.q_model_optimizer1 = th.optim.Adam(
            self.q_model1.parameters(), lr=self.cfg.policy_lr
        )
        self.q_model_optimizer2 = th.optim.Adam(
            self.q_model2.parameters(), lr=self.cfg.policy_lr
        )
        self.policy_optimizer = th.optim.Adam(
            self.policy.parameters(), lr=self.cfg.policy_lr
        )

    def _get_action_for_rollout_step(self, state):
        action = self.policy.act(state)
        noise = th.randn_like(action) * self.cfg.act_noise_scale
        return th.clip(
            action + noise,
            min=th.tensor(self.env._env.action_space.low),
            max=th.tensor(self.env._env.action_space.high),
        )

    def _get_target_policy_action(self, state):
        action = self.target_policy(state)
        noise = th.randn_like(action) * self.cfg.target_act_noise_scale
        noise = th.clip(
            noise,
            min=-self.cfg.target_act_noise_clip,
            max=self.cfg.target_act_noise_clip,
        )
        action = th.clip(
            self.policy.act(state) + noise,
            min=th.tensor(self.env._env.action_space.low),
            max=th.tensor(self.env._env.action_space.high),
        )
        return action

    def _update_q_model_step(self, transitions: Transitions) -> dict:
        states = transitions.state
        actions = transitions.action
        next_states = transitions.next_state
        rewards = transitions.reward
        terminateds = transitions.terminated.to(th.int)

        with th.no_grad():
            future_returns = th.minimum(
                self.target_q_model1(
                    next_states, self._get_target_policy_action(next_states)
                ),
                self.target_q_model2(
                    next_states, self._get_target_policy_action(next_states)
                ),
            ).squeeze(1)
        target_q_value = rewards + self.cfg.gamma * (1 - terminateds) * future_returns
        metrics = dict()
        for i, (q_model, optimizer) in enumerate(
            (
                (self.q_model1, self.q_model_optimizer1),
                (self.q_model2, self.q_model_optimizer2),
            )
        ):
            # mean squared belman error
            loss = F.mse_loss(q_model(states, actions).squeeze(1), target_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metrics[f"q_model{i}_loss"] = loss.item()
            metrics[f"q_model{i}_grad_norm"] = train_utils.compute_grad_norm(q_model)
        return metrics

    def _update_policy_step(self, transitions: Transitions) -> dict:
        states = transitions.state
        optimizer = self.policy_optimizer
        policy = self.policy
        q_model = self.q_model1
        loss = -q_model(states, policy.act(states)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return dict(
            policy_loss=loss.item(),
            policy_grad_norm=train_utils.compute_grad_norm(policy),
        )

    def _update_target_policy_step(self) -> None:
        factor = self.cfg.polyak_update_factor
        model_utils.polyak_update(self.policy, self.target_policy, factor)

    def _update_target_q_model_step(self) -> None:
        factor = self.cfg.polyak_update_factor
        model_utils.polyak_update(self.q_model1, self.target_q_model1, factor)
        model_utils.polyak_update(self.q_model2, self.target_q_model2, factor)
