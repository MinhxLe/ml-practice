from mle.rl.tabular import MRP, evaluate_policy, policy_iteration, value_iteration
from mle.rl.environment import TabularEnvSpec
import numpy as np

from tests.envs.riverswim import RiverSwim


def riverswim_to_env(rs: RiverSwim) -> TabularEnvSpec:
    return TabularEnvSpec(
        n_actions=rs.num_actions,
        n_states=rs.num_states,
        dynamics=rs.T,
        reward=rs.R,
    )


def test_mrp_evaluate_value_fn():
    mrp = MRP(
        n_actions=2,
        n_states=2,
        reward=np.array([0, 1]),
        dynamics=np.array(
            [
                [0, 1],
                [0, 1],
            ]
        ),
        gamma=0.5,
    )
    assert np.allclose(mrp.evaluate_value_fn(), np.array([1, 2]))


def test_evaluate_value_fn():
    policy = np.array(
        [
            [0, 1],
            [0, 1],
        ]
    )
    env = TabularEnvSpec(
        n_actions=2,
        n_states=2,
        reward=np.array(
            [
                [0, 0],
                [0, 1],
            ]
        ),
        dynamics=np.array(
            [
                [
                    [0, 1],
                    [0, 1],
                ],
                [
                    [0, 1],
                    [0, 1],
                ],
            ],
        ),
    )
    value = evaluate_policy(policy, env, gamma=0.5, tol=1e-5)
    assert np.allclose(value, np.array([1, 2]))


def test_evaluate_policy_riverswim():
    env = riverswim_to_env(RiverSwim("WEAK", 1234))
    # all right
    policy = np.zeros((env.n_states, env.n_actions))
    policy[:, 1] = 1

    value = evaluate_policy(policy, env, gamma=0.99, tol=1e-3)
    # from a different implementation
    target_value = np.array([30.328, 31.097, 32.336, 33.753, 35.267, 36.859])
    assert np.linalg.norm(value - target_value) < 1e-3


def test_policy_iteration_riverswim():
    env = riverswim_to_env(RiverSwim("STRONG", 1234))
    policy, v = policy_iteration(env, gamma=0.67, tol=1e-3)

    expected_policy = np.eye(2)[[0, 0, 1, 1, 1, 1]]
    expected_v = np.array([0.01515, 0.01015, 0.0093, 0.04278, 0.27441, 1.79521])

    assert np.all(policy == expected_policy)
    # this is close enough
    assert np.allclose(v, expected_v, atol=1e-2)


def test_policy_iteration_riverswim():
    env = riverswim_to_env(RiverSwim("STRONG", 1234))
    policy, v = value_iteration(env, gamma=0.67, tol=1e-3)

    expected_policy = np.eye(2)[[0, 0, 1, 1, 1, 1]]
    expected_v = np.array([0.01515, 0.01015, 0.0093, 0.04278, 0.27441, 1.79521])

    assert np.all(policy == expected_policy)
    # this is close enough
    assert np.allclose(v, expected_v, atol=1e-2)
