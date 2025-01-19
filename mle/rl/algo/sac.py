from typing import Callable
import attrs
from mle.rl.algo.off_policy import BaseOffPolicyCfg, BaseOffPolicy
from mle.rl.core import Transitions
from mle.rl.models.policy import TanhNormalPolicy
from mle.rl.models.q_model import BaseQModel
from mle.utils import model_utils, train_utils
import torch as th
from torch.nn import functional as F


@attrs.frozen
class SACCfg(BaseOffPolicyCfg):
    pass
    entropy_factor: float  # entropy factor into loss
    entropy_clip: float
    policy_lr: float
    q_model_lr: float
    polyak_update_factor: float
    update_freq: int = attrs.field(init=False, default=1)


@attrs.define
class SAC(BaseOffPolicy[SACCfg]):
    policy: TanhNormalPolicy
    policy_optimizer: th.optim.Adam = attrs.field(init=False)

    create_q_model_fn: Callable[[], BaseQModel]
    q_model1: BaseQModel = attrs.field(init=False)
    target_q_model1: BaseQModel = attrs.field(init=False)
    q_model_optimizer1: th.optim.Adam = attrs.field(init=False)

    q_model2: BaseQModel = attrs.field(init=False)
    target_q_model2: BaseQModel = attrs.field(init=False)
    q_model_optimizer2: th.optim.Adam = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.q_model1 = self.create_q_model_fn()
        self.q_model2 = self.create_q_model_fn()
        self.target_q_model1 = model_utils.copy_model(self.q_model1)
        self.target_q_model2 = model_utils.copy_model(self.q_model2)
        self.q_model_optimizer1 = th.optim.Adam(
            self.q_model1.parameters(), lr=self.cfg.policy_lr
        )
        self.q_model_optimizer2 = th.optim.Adam(
            self.q_model2.parameters(), lr=self.cfg.policy_lr
        )
        self.policy_optimizer = th.optim.Adam(
            self.policy.parameters(), lr=self.cfg.policy_lr
        )
        super().__attrs_post_init__()

    def _get_action_for_rollout_step(self, state) -> th.Tensor:
        return self.policy.act(state)

    def _update_q_model_step(self, transitions: Transitions) -> dict:
        cfg = self.cfg
        policy = self.policy
        rewards = transitions.reward
        states = transitions.state
        actions = transitions.action
        terminated = transitions.terminated.to(th.int)
        next_states = transitions.next_state

        with th.no_grad():
            next_actions = policy.act(next_states)
            next_action_log_probs = policy.log_prob(states, next_actions)
            future_returns = th.minimum(
                self.target_q_model1(next_states, next_actions),
                self.target_q_model2(next_states, next_actions),
            ).squeeze(1)
            entropy = th.clip(next_action_log_probs, -self.cfg.entropy_clip, 0)
            # entropy = 0
            target_q = rewards + cfg.gamma * (1 - terminated) * (
                future_returns - cfg.entropy_factor * entropy
            )
        metrics = dict()
        for i, (q_model, optimizer) in enumerate(
            [
                (self.q_model1, self.q_model_optimizer1),
                (self.q_model2, self.q_model_optimizer2),
            ]
        ):
            loss = F.mse_loss(q_model(states, actions).squeeze(1), target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metrics[f"q_model{i}_loss"] = loss.item()
            metrics[f"q_model{i}_grad_norm"] = train_utils.compute_grad_norm(q_model)
        return metrics

    def _update_policy_step(self, transitions: Transitions) -> dict:
        states = transitions.state
        actions = self.policy.act(states)
        log_probs = self.policy.log_prob(states, actions)
        entropy = th.clip(log_probs, -self.cfg.entropy_clip, 0)
        # entropy = 0
        q_value = th.minimum(
            self.q_model1(states, actions),
            self.q_model2(states, actions),
        )
        loss = -(q_value - self.cfg.entropy_factor * entropy).mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return dict(
            policy_loss=loss.item(),
            policy_grad_norm=train_utils.compute_grad_norm(self.policy),
            mean_policy_log_prob=log_probs.mean().item(),
            mean_policy_q_value=q_value.mean().item(),
        )

    def _update_target_policy_step(self) -> None:
        # SAC does not have a target policy
        return

    def _update_target_q_model_step(self) -> None:
        factor = self.cfg.polyak_update_factor
        model_utils.polyak_update(self.q_model1, self.target_q_model1, factor)
        model_utils.polyak_update(self.q_model2, self.target_q_model2, factor)
