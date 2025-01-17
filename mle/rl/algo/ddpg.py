import copy
import attrs
import torch
import wandb
from loguru import logger

from mle.external import wandb_driver
from mle.rl.core import Trajectory
from mle.rl.env import GymEnv
from mle.rl.metrics import MetricsTracker
from mle.rl.models.policy import BasePolicy
from torch import optim
from torch.nn import functional as F

from mle.rl.models.q_model import BaseQModel
from mle.rl.replay_buffer import ReplayBuffer
from mle.utils.train_utils import compute_grad_norm


@attrs.frozen(kw_only=True)
class DDPGCfg:
    gamma: float = 1.0
    replay_buffer_size: int = 100_000
    policy_lr: float = 1e-3
    q_model_lr: float = 1e-3
    target_update_factor: float = 0.005
    n_epochs: int = 1_000_000
    max_episode_steps: int = 1_000
    batch_size: int = 200
    action_noise: float = 0.1
    update_freq: int = 1
    update_after: int = 1_000

    # log
    eval_freq: int = 1000
    debug: bool = True

    def __attrs_post_init__(self):
        assert self.update_after >= self.batch_size


@attrs.define
class DDPG:
    policy: BasePolicy
    q_model: BaseQModel
    env: GymEnv
    cfg: DDPGCfg

    _eval_env: GymEnv = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        assert not self.env.is_discrete
        self._eval_env = copy.deepcopy(self.env)

    def _create_target(self, model):
        target = copy.deepcopy(model)
        target.load_state_dict(model.state_dict())
        return target

    def _update_target(self, target_model, model, update_factor: float):
        model_dict = model.state_dict()
        target_model_dict = target_model.state_dict()
        for key in model_dict:
            target_model_dict[key] = (model_dict[key] * update_factor) + (
                target_model_dict[key] * (1 - update_factor)
            )
        target_model.load_state_dict(target_model_dict)

    @torch.no_grad()
    def sample_trajs(self, max_episode_steps: int, n_trajs: int) -> list[Trajectory]:
        env = self._eval_env
        policy = self.policy
        trajs = []
        for _ in range(n_trajs):
            env.reset()
            traj = Trajectory()
            done = False
            while not done:
                action = policy.act(self.env.state)
                action = torch.clip(
                    action,
                    min=torch.tensor(env._env.action_space.low),
                    max=torch.tensor(env._env.action_space.high),
                )
                transition, terminated = env.step(action)
                traj.append(transition)
                done = env.t == max_episode_steps or terminated
            trajs.append(traj)
        # [TODO] this is unideal
        return trajs

    def train(self):
        cfg = self.cfg
        policy = self.policy
        q_model = self.q_model
        env = self.env
        replay_buffer = ReplayBuffer(max_size=cfg.replay_buffer_size)
        target_q_model = self._create_target(q_model)
        target_policy = self._create_target(policy)
        policy_optimizer = optim.Adam(policy.parameters(), lr=cfg.policy_lr)
        q_model_optimizer = optim.Adam(q_model.parameters(), lr=cfg.q_model_lr)

        if wandb_driver.is_initialized():
            wandb.watch(self.policy, log="all")
            wandb.watch(self.q_model, log="all")

        env.reset()
        done = False
        for i_epoch in range(cfg.n_epochs):
            if done:
                env.reset()
                done = False

            with torch.no_grad():
                action = policy.act(env.state)
                # [TODO] add clipping?
                noise = torch.clip(
                    torch.randn_like(action) * self.cfg.action_noise, -0.5, 0.5
                )
                action = torch.clip(
                    action + noise,
                    min=torch.tensor(env._env.action_space.low),
                    max=torch.tensor(env._env.action_space.high),
                )
            transition, terminated = env.step(action)
            done = (env.t == cfg.max_episode_steps) or terminated

            replay_buffer.push(transition=transition)
            metrics_tracker = MetricsTracker()

            if i_epoch >= cfg.update_after and (i_epoch % cfg.update_freq) == 0:  #
                # IMPORTANT we delay the update steps but still
                # maintain frequency of updates
                for update_step in range(cfg.update_freq):
                    transitions = replay_buffer.sample(cfg.batch_size)
                    states = transitions["state"]
                    rewards = transitions["reward"]
                    actions = transitions["action"]
                    terminated = transitions["terminated"]
                    future_returns = torch.zeros_like(rewards)
                    with torch.no_grad():
                        future_returns[~terminated] = target_q_model(
                            transitions[~terminated]["next_state"],
                            target_policy.act(transitions[~terminated]["next_state"]),
                        ).squeeze(1)
                    target_returns = rewards + (cfg.gamma * future_returns)
                    logger.debug(f"max target return: {target_returns.max().item()}")
                    metrics = dict()

                    # update q function
                    q_model_loss = F.mse_loss(q_model(states, actions), target_returns)
                    q_model_optimizer.zero_grad()
                    q_model_loss.backward()
                    q_model_optimizer.step()
                    logger.debug(f"q loss: {q_model_loss.item()}")
                    # update policy
                    policy_loss = -q_model(states, policy(states)).mean()
                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()
                    logger.debug(f"policy loss: {policy_loss.item()}")
                    if i_epoch % 100 == 0:
                        __import__("ipdb").set_trace()

                    if cfg.debug:
                        metrics["q_model_loss"] = q_model_loss.item()
                        metrics["q_model_grad_norm"] = compute_grad_norm(q_model)
                        metrics["policy_loss"] = policy_loss.item()
                        metrics["policy_grad_norm"] = compute_grad_norm(policy)

                    self._update_target(
                        target_q_model, q_model, cfg.target_update_factor
                    )
                    self._update_target(target_policy, policy, cfg.target_update_factor)

                    if wandb_driver.is_initialized():
                        wandb.log(metrics | dict(epoch=i_epoch + 1))

            if ((i_epoch + 1) % cfg.eval_freq) == 0:
                trajs = self.sample_trajs(cfg.max_episode_steps, 20)
                metrics = metrics_tracker.capture(trajs)
                if wandb_driver.is_initialized():
                    wandb.log(metrics | dict(epoch=i_epoch + 1))
                logger.info(
                    f"Epoch {i_epoch+1}/{cfg.n_epochs}, mean_reward: {metrics['mean_traj_reward']:.3f}"
                )
