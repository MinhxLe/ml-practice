from dataclasses import dataclass
import numpy as np


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
