import torch
from mle.rl.replay_buffer import ReplayBuffer
from mle.rl import utils


def build_transition(
    state=0,
    action=1,
    reward=2,
    next_state=1,
    time_step=1,
):
    return utils.build_transition(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        time_step=time_step,
    )


def test_basic():
    t = build_transition(
        state=0,
        action=1,
        reward=2,
        next_state=1,
        time_step=1,
    )
    rb = ReplayBuffer(10)
    rb.push(t)
    sampled_ts = rb.sample(n=1)
    assert sampled_ts.shape == torch.Size([1])

    sampled_t = sampled_ts[0]
    assert sampled_t["state"] == 0
    assert sampled_t["action"] == 1
    assert sampled_t["reward"] == 2
    assert sampled_t["next_state"] == 1
    assert sampled_t["time_step"] == 1


def test_max_size():
    rb = ReplayBuffer(1)
    t1 = build_transition(time_step=0)
    t2 = build_transition(time_step=1)

    rb.push(t1)
    rb.push(t2)
    assert rb.sample()[0]["time_step"] == 1


def test_sample_multiple():
    rb = ReplayBuffer(5)
    rb.push(build_transition())
    rb.push(build_transition())
    sampled_ts = rb.sample(n=2)
    assert sampled_ts.shape[0] == 2
