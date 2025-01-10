"""
Tabular MDP evaluation and policy improvement.
"""

from typing import Tuple
import numpy as np
from dataclasses import dataclass
from loguru import logger
from einops import einsum, reduce
from mle.rl.environment import TabularEnvSpec


@dataclass
class MRP:
    n_actions: int
    n_states: int
    dynamics: np.ndarray  # dynamics[s] is transition probabilities
    reward: np.ndarray  # reward[s] is reward for being in a state
    gamma: float

    def __post_init__(self):
        assert 0 < self.gamma <= 1.0

        from_state_dim, to_state_dim = self.dynamics.shape
        assert from_state_dim == self.n_states
        assert to_state_dim == self.n_states
        assert np.all(self.dynamics.sum(axis=-1) == 1)

        (state_dim,) = self.reward.shape
        assert state_dim == self.n_states

    def evaluate_value_fn(self, max_iterations: int = 10_000) -> np.ndarray:
        """
        returns expected return per state using bellman backup
        """
        prev_value = None
        value = np.zeros(self.n_states)
        for _ in range(max_iterations):
            prev_value = value
            value = self.reward + self.gamma * (
                einsum(self.dynamics, value, "i j, j -> i")
            )
            # value = self.reward + self.gamma * self.dynamics @ value
            if np.allclose(prev_value, value):
                break

        assert prev_value is not None
        if not np.allclose(prev_value, value):
            logger.warning("value function did not converge before max_iterations")
        return value


def _validate_gamma(gamma: float):
    assert 0 <= gamma < 1


def _validate_policy(policy: np.ndarray):
    assert np.all(policy.sum(axis=-1) == 1)


def _validate_policy_env(policy: np.ndarray, env: TabularEnvSpec):
    state_dim, action_dim = policy.shape
    assert state_dim == env.n_states
    assert action_dim == env.n_actions
    assert np.all(policy.sum(axis=-1) == 1)


def evaluate_policy(
    policy: np.ndarray,
    env: TabularEnvSpec,
    gamma: float,
    tol: float = 1e-3,
    max_iters: int = 1_000,
) -> np.ndarray:
    """Bellman backup to calculate value given policy"""
    _validate_gamma(gamma)
    _validate_policy(policy)
    _validate_policy_env(policy, env)
    prev_value = None
    value = np.zeros(env.n_states)

    for _ in range(max_iters):
        prev_value = value
        q_value = env.reward + gamma * (
            einsum(
                env.dynamics,
                value,
                "from_state action to_state, to_state -> from_state action",
            )
        )
        value = einsum(
            q_value,
            policy,
            "from_state action, from_state action -> from_state",
        )
        if np.linalg.norm(prev_value - value, np.inf) < tol:
            break

    assert prev_value is not None
    if np.linalg.norm(prev_value - value, np.inf) < tol:
        logger.warning("value function did not converge before max_iterations")
    return value


def improve_policy(
    policy: np.ndarray,
    v_policy: np.ndarray,
    env: TabularEnvSpec,
    gamma: float,
    tol: float = 1e-3,
    max_iters: int = 1_000,
) -> np.ndarray:
    """
    policy improvement algo
    """
    q = env.reward + gamma * einsum(env.dynamics, v_policy, "fs a ts,ts -> fs a")
    # returns a deterministic policy
    return np.eye(env.n_actions)[np.argmax(q, axis=1)]


def policy_iteration(
    env: TabularEnvSpec, gamma: float, tol: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    given an env, use policy iteration got find optimal policy and value function
    returns: policy and value
    """
    policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    while True:
        old_policy = policy
        v = evaluate_policy(policy, env, gamma, tol)
        policy = improve_policy(policy, v, env, gamma)
        if np.linalg.norm(policy - old_policy, np.inf) < tol:
            break
    return policy, v


def value_iteration(
    env: TabularEnvSpec, gamma: float, tol: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    v = np.zeros((env.n_states,))
    while True:
        old_v = v
        q = env.reward + gamma * einsum(env.dynamics, old_v, "fs a ts,ts->fs a")
        # v = np.max(q, axis=1)
        v = reduce(q, "s a->s", "max")
        if np.linalg.norm(v - old_v, np.inf) < tol:
            break

    q_star = env.reward + gamma * einsum(env.dynamics, old_v, "fs a ts,ts->fs a")
    policy = np.eye(env.n_actions)[np.argmax(q_star, axis=1)]
    return policy, v


def every_mc_estimate(trajectory: np.ndarray, n_states: int, gamma: float):
    """
    trajectory: np.array (time, (state, action, reward))
    returns:
    value function
    """
    n_steps, _ = trajectory.shape

    states = trajectory[:, 0]
    rewards = trajectory[:, 2]

    v = np.zeros((n_states,))
    n = np.zeros((n_states,))
    g = 0
    for i in reversed(range(n_steps)):
        s_i = states[i]
        g = rewards[i] + gamma * g
        n[s_i] += 1
        # runnin average
        # v[s_i] = v[s_i] * (n[s_i] - 1) / n[s_i] + g / n[s_i]
        v[s_i] = v[s_i] + (g - v[s_i]) / n[s_i]
    return v


def incremental_mc_estimation(
    v: np.ndarray,
    trajectory: np.ndarray,
    gamma: float,
    lr: float,
) -> np.ndarray:
    """
    trajectory: np.array (time, (state, action, reward))
    returns:
    value function
    """
    n_steps, _ = trajectory.shape

    states = trajectory[:, 0]
    rewards = trajectory[:, 2]

    g = 0
    for i in reversed(range(n_steps)):
        s_i = states[i]
        g = rewards[i] + gamma * g
        v[s_i] = v[s_i] + lr * (g - v[s_i])
    return v


def td_update(
    v: np.ndarray,
    trajectory: np.ndarray,
    n_steps: int,
    gamma: float,  # return decay rate
    lr: float,
) -> np.ndarray:
    """
    parameters:
    v: value function
    trajectory: (time, (state, action, reward))
    """
    trajectory_length, _ = trajectory.shape

    states = trajectory[:, 0].astype(int)
    rewards = trajectory[:, 2]
    gamma_powers = gamma ** np.arange(n_steps)
    for i in range(trajectory_length):
        steps_left = min(n_steps, trajectory_length - i)
        s_i = states[i]
        discounted_reward = (
            rewards[i : i + steps_left] * gamma_powers[:steps_left]
        ).sum()
        if i + steps_left < trajectory_length:
            bootstrap = (gamma**steps_left) * v[i + n_steps]
        else:
            bootstrap = 0
        target = discounted_reward + bootstrap
        v[s_i] = v[s_i] + lr * (target - v[s_i])
    return v
