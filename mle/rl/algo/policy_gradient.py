from typing import Callable
import torch
from torch import optim, nn
from mle.rl.env import GymEnv
from mle.rl.metrics import MetricsTracker
from mle.rl.models.policy import BasePolicy
from mle.rl.core import Trajectory, calculate_returns
from mle.trainer import BaseTrainer
from mle.external import wandb_driver
from mle.utils import train_utils
import wandb
from loguru import logger
import attrs


@attrs.frozen(kw_only=True)
class PolicyGradientCfg:
    gamma: float
    # training
    lr: float
    n_epochs: int
    max_episode_steps: int
    batch_size: int
    # log
    train_log_freq: int = 1
    debug: bool = True


class PolicyTrainer(BaseTrainer):
    def __init__(
        self,
        policy: BasePolicy,
        baseline: nn.Module | None,
        lr: float,
        gamma: float,
    ):
        super().__init__(policy, optim.Adam, dict(lr=lr))
        self.baseline = baseline
        self.gamma = gamma

    def _calculate_advantages(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
    ):
        if self.baseline is not None:
            with torch.no_grad():
                advantages = returns - self.baseline(states).squeeze(1)
        else:
            advantages = returns

        mu = torch.mean(advantages)
        std = torch.std(advantages)
        return (advantages - mu) / std

    def loss_fn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ):
        action_log_probs = self.model.action_dist(states).log_prob(actions)
        return -(action_log_probs * advantages).mean()

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
        return self._step_optimizer(
            states=all_states, advantages=all_advantages, actions=all_actions
        )


class BaselineTrainer(BaseTrainer):
    def __init__(
        self,
        baseline: nn.Module,
        lr: float,
        gamma: float,
        n_gradient_steps: int,
    ):
        super().__init__(baseline, optim.Adam, dict(lr=lr))
        self.gamma = gamma
        self.n_gradient_steps = n_gradient_steps

    def loss_fn(self, states: torch.Tensor, returns: torch.Tensor):
        return nn.MSELoss()(self.model(states), returns)

    def update(self, trajs: list[Trajectory]) -> float:
        traj_tds = [t.to_tensordict() for t in trajs]
        all_states = torch.concat([t["state"] for t in traj_tds])
        all_returns = torch.concat(
            [calculate_returns(traj, self.gamma) for traj in trajs]
        )
        # baseline can be updated multiple times since
        for _ in range(self.n_gradient_steps):
            return self._step_optimizer(states=all_states, returns=all_returns)


class PolicyGradient:
    def __init__(
        self,
        create_policy_fn: Callable[[], BasePolicy],
        create_baseline_fn: Callable[[], nn.Module] | None,
        env: GymEnv,
        cfg: PolicyGradientCfg,
    ):
        self.cfg = cfg
        self.create_policy_fn = create_policy_fn
        self.create_baseline_fn = create_baseline_fn
        self.env = env

        self.policy = create_policy_fn()
        if self.create_baseline_fn:
            self.baseline = self.create_baseline_fn()
        else:
            self.baseline = None
        self.metrics_tracker = MetricsTracker()
        self._validate_policy_env(self.policy, self.env)

    def _validate_policy_env(self, policy: BasePolicy, env: GymEnv):
        assert policy.state_dim == env.state_dim
        assert policy.action_dim == env.action_dim

    def _init_policy_trainer(self) -> PolicyTrainer:
        return PolicyTrainer(
            self.policy,
            lr=self.cfg.lr,
            baseline=self.baseline,
            gamma=self.cfg.gamma,
        )

    def _init_baseline_trainer(self) -> BaselineTrainer | None:
        cfg = self.cfg
        if self.baseline:
            baseline_trainer = BaselineTrainer(
                self.baseline,
                lr=cfg.lr,
                gamma=cfg.gamma,
                n_gradient_steps=1,
            )
        else:
            baseline_trainer = None
        return baseline_trainer

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

    def get_returns(self, traj: Trajectory, gamma: float) -> torch.Tensor:
        returns = []
        current_return = 0.0
        for transition in reversed(traj):
            current_return = transition.reward + gamma * current_return
            returns.append(current_return)
        return torch.tensor(list(reversed(returns)), dtype=torch.float32)

    def train(self) -> None:
        cfg = self.cfg

        policy_trainer = self._init_policy_trainer()
        baseline_trainer = self._init_baseline_trainer()

        if wandb_driver.is_initialized():
            wandb.watch(self.policy, log="all")
            if self.baseline:
                wandb.watch(self.baseline, log="all")

        metrics_tracker = MetricsTracker()
        for i_epoch in range(cfg.n_epochs):
            trajs = self.sample_trajs(
                self.policy,
                self.env,
                max_episode_steps=cfg.max_episode_steps,
                max_total_steps=cfg.batch_size,
            )
            if baseline_trainer is not None:
                baseline_loss = baseline_trainer.update(trajs=trajs)
            else:
                baseline_loss = None
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
