from dataclasses import dataclass
from typing import Tuple
import numpy as np
import gymnasium as gym
import torch
from loguru import logger

from mle.rl.utils import Transition, build_transition


@dataclass(frozen=True)
class TabularEnvSpec:
    n_actions: int
    n_states: int
    dynamics: np.ndarray  # dynamics[s, a] is transition probabilities
    reward: np.ndarray  # reward[s, a] is reward for being in a state
    terminal_states: list[int] | None = None

    def __post_init__(self):
        from_state_dim, from_action_dim, to_state_dim = self.dynamics.shape
        assert from_state_dim == self.n_states
        assert to_state_dim == self.n_states
        assert from_action_dim == self.n_actions
        assert np.all(self.dynamics.sum(axis=-1) == 1)

        state_dim, action_dim = self.reward.shape
        assert state_dim == self.n_states
        assert action_dim == self.n_actions


step_dtype = np.dtype(
    [
        ("state", np.int32),
        ("action", np.int32),
        ("reward", np.float32),
        ("next_state", np.int32),
        ("idx", np.int32),
    ]
)


@dataclass
class StepState:
    state: int
    action: int
    reward: float
    next_state: int
    idx: int

    def to_np(self) -> np.void:
        return np.void(
            (self.state, self.action, self.reward, self.next_state, self.idx),
            dtype=step_dtype,
        )


@dataclass
class TabularEnv:
    spec: TabularEnvSpec
    initial_state: int
    seed: int

    def __post_init__(self):
        np.random.seed(self.seed)
        self.state = self.initial_state
        self.t = 0

    def reset(self) -> None:
        self.state = self.initial_state
        self.t = 0

    def is_terminal(self) -> bool:
        return (
            self.spec.terminal_states is not None
            and self.state in self.spec.terminal_states
        )

    def step(self, action: int | np.ndarray) -> StepState:
        if self.is_terminal():
            raise ValueError("env in terminal state, please reset")

        n_actions = self.spec.n_actions
        if isinstance(action, int):
            assert 0 <= action < n_actions
            selected_action = action
        else:
            selected_action = np.random.choice(range(n_actions), p=action)

        next_state = np.random.choice(
            range(self.spec.n_states),
            p=self.spec.dynamics[self.state, selected_action],
        )
        reward = self.spec.reward[self.state, selected_action]

        state = StepState(self.state, selected_action, reward, next_state, self.t)
        self.t += 1
        self.state = next_state
        return state


class GymEnv:
    def __init__(self, env: gym.Env, torch_type=torch.float32):
        self._env = env
        self.t = 0
        self.terminated = False
        self.reset()
        self.torch_type = torch_type

    @property
    def is_discrete(self):
        return isinstance(self._env.action_space, gym.spaces.Discrete)

    @property
    def action_dim(self) -> int:
        if self.is_discrete:
            return self._env.action_space.n.item()
        else:
            # [TODO] this is not perfect
            return self._env.action_space.shape[0]

    @property
    def state_dim(self) -> int:
        return self.state.shape[0]

    @property
    def state(self):
        env = self._env.unwrapped
        if hasattr(env, "state"):
            state = env.state
        elif hasattr(env, "state_vector"):
            state = env.state_vector()
        else:
            raise NotImplementedError
        return torch.tensor(state, dtype=self.torch_type)

    def reset(self):
        self.t = 0
        self.terminated = False
        self._env.reset()

    def step(self, action) -> Tuple[Transition, bool]:
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        if self.terminated:
            raise ValueError("environment terminated, please reset!")
        state = self.state
        next_state, reward, terminated, truncated, _ = self._env.step(action)
        transition = build_transition(
            state,  # already casted to torch
            action,
            reward,
            torch.tensor(next_state, dtype=self.torch_type) if not terminated else None,
            self.t,
            terminated=terminated,
        )
        self.t += 1
        self.terminated = terminated
        if truncated:
            logger.warning("truncated")
        return transition, terminated or truncated
