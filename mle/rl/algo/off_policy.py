"""
base for off policy algos
"""

import pprint
import abc
from typing import Generic, TypeVar, final
import attrs
import copy
from loguru import logger
import torch as th
import wandb
from mle.external import wandb_driver
from mle.rl.core import Trajectory, Transitions
from mle.rl.env import GymEnv
from mle.rl.metrics import MetricsTracker
from mle.rl.models.policy import BasePolicy
from mle.rl.replay_buffer import ReplayBuffer


@attrs.frozen(kw_only=True)
class BaseOffPolicyCfg:
    gamma: float
    replay_buffer_size: int = 100_000

    # training args
    n_epochs: int
    n_steps_per_epoch: int
    max_episode_steps: int
    update_freq: int
    update_after: int
    batch_size: int
    act_noise_scale: float
    policy_update_freq: int

    debug: bool = True
    log_train_freq: int = 1000
    n_eval_trajs: int = 50

    def __attrs_post_init__(self):
        assert self.update_after >= self.batch_size


CfgT = TypeVar("CfgT", bound=BaseOffPolicyCfg)


@attrs.define(kw_only=True)
class BaseOffPolicy(abc.ABC, Generic[CfgT]):
    policy: BasePolicy
    env: GymEnv
    cfg: CfgT
    replay_buffer: ReplayBuffer = attrs.field(init=False)
    _eval_env: GymEnv = attrs.field(init=False)

    def __attrs_post_init__(self):
        cfg = self.cfg
        assert not self.env.is_discrete
        self._eval_env = copy.deepcopy(self.env)
        self.replay_buffer = ReplayBuffer(max_size=cfg.replay_buffer_size)

    @abc.abstractmethod
    def _update_q_model_step(self, transitions: Transitions) -> dict: ...

    @abc.abstractmethod
    def _update_policy_step(self, transitions: Transitions) -> dict: ...

    @abc.abstractmethod
    def _update_target_step(self) -> None: ...

    @th.no_grad()
    def sample_eval_trajs(self, n) -> list[Trajectory]:
        env = self._eval_env
        policy = self.policy
        trajs = []
        for _ in range(n):
            i, done = 0, False
            traj = Trajectory()
            env.reset()
            while not done:
                i += 1
                action = policy.act(env.state)
                transition, terminated = env.step(action)
                done = terminated or i == self.cfg.max_episode_steps
                traj.append(transition)
            trajs.append(traj)
        return trajs

    @final
    def train(self):
        cfg = self.cfg
        env = self.env
        metrics_tracker = MetricsTracker()
        for i_epoch in range(cfg.n_epochs):
            env.reset()
            done = False
            traj = Trajectory()
            for i_step in range(cfg.n_steps_per_epoch):
                if done:
                    traj = Trajectory()
                    env.reset()
                    done = False
                with th.no_grad():
                    # in the sampling phase, we always use the
                    # current policy.
                    action = self.policy.act(env.state)
                    noise = th.randn_like(action) * cfg.act_noise_scale
                    action = th.clip(
                        action + noise,
                        min=th.tensor(self.env._env.action_space.low),
                        max=th.tensor(self.env._env.action_space.high),
                    )
                transition, terminated = env.step(action)
                done = env.t == cfg.max_episode_steps or terminated
                traj.append(transition)
                self.replay_buffer.push(transition)
                if i_step >= cfg.update_after and i_step % cfg.update_freq == 0:
                    for _ in range(cfg.update_freq):
                        transitions = self.replay_buffer.sample(cfg.batch_size)
                        q_model_metrics = self._update_q_model_step(transitions)
                        if i_step % cfg.policy_update_freq == 0:
                            policy_metrics = self._update_policy_step(transitions)
                        else:
                            policy_metrics = dict()
                            self._update_target_step()

                    if cfg.debug:
                        train_metrics = (
                            q_model_metrics
                            | policy_metrics
                            | dict(
                                epoch=i_epoch + 1,
                                step=i_epoch * cfg.n_steps_per_epoch + i_step,
                            )
                        )
                        if i_step % cfg.log_train_freq == 0:
                            logger.debug(pprint.pformat(train_metrics))
                        if wandb_driver.is_initialized():
                            wandb.log(train_metrics)
            # [TODO] we probably want to capture actions w/o noise
            trajs = self.sample_eval_trajs(self.cfg.n_eval_trajs)
            metrics = metrics_tracker.capture(trajs)
            if wandb_driver.is_initialized():
                wandb.log(metrics | dict(epoch=i_epoch + 1))
            logger.info(f"Epoch {i_epoch+1}/{cfg.n_epochs}\n{pprint.pformat(metrics)}")
