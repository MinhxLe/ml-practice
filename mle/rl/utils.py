from numpy import isin
from tensordict import TensorDict
import torch

Transition = TensorDict


def build_transition(
    state,
    action,
    reward,
    next_state,
    time_step,
    terminated=False,
) -> Transition:
    if not terminated:
        assert next_state is not None
    else:
        assert next_state is None
        # we want a placeholder value for next_state
        # so that the TensorDict only contains tensors
        next_state = torch.zeros(state.shape)

    return Transition(
        dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            time_step=time_step,
            terminated=terminated,
        ),
        # [NOTE] it is easier to think of everything as having a batch_dim.
    )
