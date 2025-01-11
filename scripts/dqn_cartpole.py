"""
DQN implementation for CartPool

hyperparameters from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from sys import exec_prefix
import wandb
import torch
import gymnasium as gym
from mle.utils import init_project
from mle.scheduler import ExponentialDecayScheduler
from mle.config import BaseCfg
from mle.rl.replay_buffer import ReplayBuffer
from mle.rl.environment import GymEnv
from torch import nn, optim
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass(frozen=True)
class Cfg(BaseCfg):
    project_name: str = "dqn_cartpole"

    # rl configuration
    # reward decay
    gamma: float = 0.99
    # eps greedy policy
    start_eps: float = 0.9
    end_eps: float = 0.05
    eps_decay_rate: float = 1e-3

    target_update_interval = 100
    tau: float = 0.005  # soft update of weights

    # optimizer
    lr: float = 1e-4

    # training config
    batch_size: int = 128
    replay_buffer_size: int = 10_000
    n_episodes: int = 600

    # log config
    train_log_interval: int = 1_000


class Model(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, state):
        return self.layers(state)


def select_eps_greedy_action(q_model: Model, state, eps: float):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)

    if len(state.shape) == 1:
        batched = False
        state = state.unsqueeze(0)
    else:
        assert len(state.shape) == 2
        batched = True

    with torch.no_grad():
        q = q_model(state)
    batch_size, n_actions = q.shape
    greedy_action = q.argmax(dim=1)
    random_action = torch.randint(low=0, high=n_actions, size=(batch_size,))
    action = torch.where(
        torch.rand(batch_size) < eps,
        random_action,
        greedy_action,
    )
    if batched:
        # [TODO] maybe we want to return it to numpy
        return action
    else:
        return action.item()


def get_target_q(target_model, transitions, gamma: float):
    """
    TD(0)
    """
    (batch_size,) = transitions.shape
    rewards = transitions["reward"]
    terminated = transitions["terminated"]
    next_state_values = torch.zeros(batch_size)
    with torch.no_grad():
        bootstrap_q = (
            target_model(transitions[~terminated]["next_state"]).max(axis=1).values
        )
        next_state_values[~terminated] = bootstrap_q
    return rewards + (gamma * next_state_values)


def update_target_model_step(model, target_model, tau: float):
    model_dict = model.state_dict()
    target_model_dict = target_model.state_dict()
    for key in model_dict:
        target_model_dict[key] = model_dict[key] * tau + target_model_dict[key] * (
            1 - tau
        )
    target_model.load_state_dict(target_model_dict)


def train_model_step(model, target_model, transitions):
    model.train()
    loss_fn = nn.SmoothL1Loss()
    q = (
        model(transitions["state"])
        .gather(1, transitions["action"].unsqueeze(1))
        .squeeze(dim=1)
    )
    target_q = get_target_q(
        target_model,
        transitions,
        cfg.gamma,
    )
    optimizer.zero_grad()
    loss = loss_fn(q, target_q)
    loss.backward()
    # nn.utils.clip_grad_value_(model.parameters(), 100)
    optimizer.step()
    return loss.item()


cfg = Cfg(log_wandb=True)
init_project(cfg)
if cfg.log_wandb:
    wandb.init(
        project=cfg.project_name,
        config=asdict(cfg),
    )


env = GymEnv(gym.make("CartPole-v1"))

model = Model(env.state_dim, env.n_actions)
target_model = Model(env.state_dim, env.n_actions)
target_model.load_state_dict(model.state_dict())

# [TODO] awkward we have to put this here
wandb.watch(model, target_model)

optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, amsgrad=True)
replay_buffer = ReplayBuffer(cfg.replay_buffer_size)
eps_scheduler = ExponentialDecayScheduler(
    cfg.start_eps, cfg.end_eps, cfg.eps_decay_rate
)
i = 0
durations = []
for i_episode in range(cfg.n_episodes):
    env.reset()
    while True:
        action = select_eps_greedy_action(model, env.state, eps_scheduler.value)
        transition, done = env.step(action)
        replay_buffer.push(transition)
        if len(replay_buffer) >= cfg.batch_size:
            transitions = replay_buffer.sample(cfg.batch_size)
            loss = train_model_step(model, target_model, transitions)
            if (i % cfg.train_log_interval) == 0:
                logger.info(f"Episode {i_episode}/{cfg.n_episodes}, loss: {loss}")

            wandb.log(dict(i=i, train_loss=loss, eps=eps_scheduler.value))
        update_target_model_step(model, target_model, cfg.tau)
        i += 1
        eps_scheduler.step()
        if done:
            break
    wandb.log(dict(episode=i_episode, duration=env.t))
    durations.append(env.t)
    i_episode += 1

wandb.finish()
