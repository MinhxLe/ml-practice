import torch
from torch import optim, nn
from mle.rl.env import GymEnv
from mle.rl.metrics import MetricsTracker
from mle.rl.models.policy import BasePolicy
from mle.rl.core import Trajectory, calculate_returns
from mle.external import wandb_driver
from mle.utils import train_utils
from typing import TypeVar, Generic
import wandb
from loguru import logger
import attrs
import abc


class BasePolicyTrainer(abc.ABC):
    def __init__(
        self,
        policy: BasePolicy,
        baseline: nn.Module,
        lr: float,
        gamma: float,
    ):
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    @abc.abstractmethod
    def update(self, trajs: list[Trajectory]) -> float: ...


class BaselineTrainer:
    def __init__(
        self,
        baseline: nn.Module,
        lr: float,
        gamma: float,
        n_steps: int,
    ):
        self.baseline = baseline
        self.gamma = gamma
        self.n_steps = n_steps
        self.optimizer = optim.Adam(baseline.parameters(), lr=lr)

    def loss_fn(self, states: torch.Tensor, returns: torch.Tensor):
        return nn.MSELoss()(self.baseline(states), returns)

    def update(self, trajs: list[Trajectory]) -> float:
        traj_tds = [t.to_tensordict() for t in trajs]
        all_states = torch.concat([t["state"] for t in traj_tds])
        all_returns = torch.concat(
            [calculate_returns(traj, self.gamma) for traj in trajs]
        )
        # baseline can be updated multiple times
        total_loss = 0
        for _ in range(self.n_steps):
            self.optimizer.zero_grad()
            loss = self.loss_fn(states=all_states, returns=all_returns)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / self.n_steps


@attrs.frozen(kw_only=True)
class BasePolicyGradientCfg:
    gamma: float
    # training
    lr: float
    n_epochs: int
    max_episode_steps: int
    batch_size: int
    # log
    train_log_freq: int = 1
    debug: bool = True


CfgT = TypeVar("CfgT", bound=BasePolicyGradientCfg)


class BasePolicyGradient(abc.ABC, Generic[CfgT]):
    def __init__(
        self,
        policy: BasePolicy,
        baseline: nn.Module,
        env: GymEnv,
        cfg: CfgT,
    ):
        self.cfg = cfg
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.metrics_tracker = MetricsTracker()
        self._validate_policy_env(self.policy, self.env)

    def _validate_policy_env(self, policy: BasePolicy, env: GymEnv):
        assert policy.state_dim == env.state_dim
        assert policy.action_dim == env.action_dim

    @abc.abstractmethod
    def _init_policy_trainer(self) -> BasePolicyTrainer: ...

    def _init_baseline_trainer(self) -> BaselineTrainer:
        assert self.baseline is not None
        return BaselineTrainer(
            self.baseline,
            lr=self.cfg.lr,
            gamma=self.cfg.gamma,
            n_steps=1,
        )

    @torch.no_grad
    def sample_trajs(
        self,
        policy,
        env: GymEnv,
        max_episode_steps: int,
        max_total_steps: int,
    ) -> list[Trajectory]:
        done = False
        i_step = 0
        trajs = []
        while i_step < max_total_steps:
            env.reset()
            traj = Trajectory()
            done = False
            while not done:
                i_step += 1
                action = policy.act(self.env.state)
                transition, terminated = env.step(action)
                traj.append(transition)
                done = (
                    env.t == max_episode_steps
                    or terminated
                    or i_step == max_total_steps
                )
            trajs.append(traj)
        return trajs

    def train(self) -> None:
        cfg = self.cfg

        policy_trainer = self._init_policy_trainer()
        baseline_trainer = self._init_baseline_trainer()

        if wandb_driver.is_initialized():
            wandb.watch(self.policy, log="all")
            wandb.watch(self.baseline, log="all")

        metrics_tracker = MetricsTracker()
        for i_epoch in range(cfg.n_epochs):
            trajs = self.sample_trajs(
                self.policy,
                self.env,
                max_episode_steps=cfg.max_episode_steps,
                max_total_steps=cfg.batch_size,
            )
            # this step is what is different between variants of policy gradient methods
            baseline_loss = baseline_trainer.update(trajs=trajs)
            policy_loss = policy_trainer.update(trajs=trajs)
            metrics = metrics_tracker.capture(trajs)
            if cfg.debug:
                metrics |= dict(
                    policy_train_loss=policy_loss,
                    policy_grad_norm=train_utils.compute_grad_norm(self.policy),
                )
                if baseline_loss is not None:
                    assert self.baseline is not None
                    metrics |= dict(
                        baseline_train_loss=baseline_loss,
                        baseline_grad_norm=train_utils.compute_grad_norm(self.baseline),
                    )

            if wandb_driver.is_initialized():
                wandb.log(metrics | dict(epoch=i_epoch + 1))
            if (i_epoch % cfg.train_log_freq) == 0:
                logger.info(
                    f"Epoch {i_epoch+1}/{cfg.n_epochs}, mean_reward: {metrics['mean_traj_reward']:.3f}"
                )
