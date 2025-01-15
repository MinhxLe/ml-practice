import abc
from typing import Callable
import torch
from dataclasses import dataclass
from mle.rl.environment import GymEnv
from mle.rl.models.policy import BasePolicy
from mle.rl.utils import Trajectory
import wandb
from loguru import logger
import numpy as np

from mle.utils import train_utils


@dataclass(frozen=True)
class PolicyGradientCfg:
    gamma: float
    # training
    lr: float
    n_epochs: int
    max_episode_steps: int
    batch_size: int

    # log
    train_log_freq: int = 10
    log_wandb: bool = True


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.traj_rewards = []

    def update(self, traj: Trajectory):
        self.traj_rewards.append(torch.sum(traj.to_tensordict()["reward"]).cpu().item())

    def get(self):
        return dict(
            mean_traj_reward=np.mean(self.traj_rewards),
        )


class PolicyGradient(abc.ABC):
    def __init__(
        self,
        create_policy_fn: Callable[[], BasePolicy],
        env: GymEnv,
        cfg: PolicyGradientCfg,
    ):
        self.cfg = cfg
        self.create_policy_fn = create_policy_fn
        self.env = env

        self.policy = create_policy_fn()
        self._validate_policy_env(self.policy, self.env)

    def _validate_policy_env(self, policy: BasePolicy, env: GymEnv):
        assert policy.state_dim == env.state_dim
        assert policy.action_dim == env.action_dim

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

    def calculate_advantages(
        self, traj: Trajectory, returns: torch.Tensor
    ) -> torch.Tensor:
        mu = torch.mean(returns)
        std = torch.std(returns)
        return (returns - mu) / std

    def update_policy(
        self,
        optimizer: torch.optim.Optimizer,
        policy: BasePolicy,
        actions: torch.Tensor,
        states: torch.Tensor,
        advantages: torch.Tensor,
    ) -> float:
        policy.train()
        optimizer.zero_grad()
        action_log_probs = policy.action_dist(states).log_prob(actions)
        loss = -(action_log_probs * advantages).mean()
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(
        self,
        reinit_policy: bool = False,
    ) -> None:
        if reinit_policy:
            self.policy = self.create_policy_fn()
        cfg = self.cfg
        policy = self.policy
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        env = self.env
        if cfg.log_wandb:
            wandb.watch(policy, log="all")

        metrics_tracker = MetricsTracker()
        for i_epoch in range(cfg.n_epochs):
            trajs = self.sample_trajs(
                policy,
                env,
                max_episode_steps=cfg.max_episode_steps,
                max_total_steps=cfg.batch_size,
            )
            traj_tds = [t.to_tensordict() for t in trajs]
            all_states = torch.concat([t["state"] for t in traj_tds])
            all_actions = torch.concat([t["action"] for t in traj_tds])
            all_advantages = []
            for traj in trajs:
                returns = self.get_returns(traj, cfg.gamma)
                advantages = self.calculate_advantages(traj, returns)
                all_advantages.append(advantages)
            all_advantages = torch.concat(all_advantages)

            loss = self.update_policy(
                optimizer,
                policy,
                actions=all_actions,
                states=all_states,
                advantages=all_advantages,
            )
            for traj in trajs:
                metrics_tracker.update(traj)
            metrics = metrics_tracker.get() | dict(
                train_loss=loss,
                grad_norm=train_utils.compute_grad_norm(policy),
                epoch=i_epoch + 1,
            )
            if cfg.log_wandb:
                wandb.log(metrics)
            if (i_epoch % cfg.train_log_freq) == 0:
                metrics_str = ", ".join(
                    [f"{mkey}: {mval:.3f}" for mkey, mval in metrics.items()]
                )
                logger.info(f"Epoch {i_epoch+1}/{cfg.n_epochs}, {metrics_str}")
