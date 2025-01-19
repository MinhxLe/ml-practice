import attrs
import torch
from typing import Callable
from mle.rl.algo.off_policy import BaseOffPolicy, BaseOffPolicyCfg
from mle.rl.models.policy import BasePolicy
from mle.rl.models.q_model import BaseQModel
from mle.utils import model_utils


@attrs.frozen
class TD3Cfg(BaseOffPolicyCfg):
    polyak_update_factor: float  # how much to update the target network

    noise_scale: float
    noise_clip: float

    policy_lr: float
    q_model_lr: float


@attrs.define
class TD3(BaseOffPolicy[TD3Cfg]):
    create_q_model_fn: Callable[[], BaseQModel]
    target_policy: BasePolicy
    target_q_model: BaseQModel

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.q_model1 = self.create_q_model_fn()
        self.q_model2 = self.create_q_model_fn()
        self.target_q_model1 = model_utils.copy_model(self.q_model1)
        self.target_q_model2 = model_utils.copy_model(self.q_model2)
        self.target_policy = model_utils.copy_model(self.policy)

    def _sample_action(self, state):
        noise = (
            torch.randn_like(self.policy.action_space.sample()) * self.cfg.noise_scale
        )
        noise = torch.clip(noise, min=-self.cfg.noise_clip, max=self.cfg.noise_clip)
        action = self.policy.act(state) + noise
        action = torch.clip(
            action,
            min=self.env._env.action_space.low,
            max=self.env._env.action_space.high,
        )
        return action

    def _update_policy_step(self, transitions):
        pass
