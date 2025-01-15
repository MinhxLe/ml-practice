from mle.rl.env import TabularEnvSpec, TabularEnv
import numpy as np


def test_step():
    spec = TabularEnvSpec(
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
                    [1, 0],
                    [0, 1],
                ],
                [
                    [1, 0],
                    [1, 0],
                ],
            ],
        ),
    )
    env = TabularEnv(spec, initial_state=1, seed=42)
    step = env.step(1)
    assert step.state == 1
    assert step.reward == 1
    assert step.next_state == 0
    assert step.idx == 0
