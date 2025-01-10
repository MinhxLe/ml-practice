from tensordict import TensorDict


def build_transition(
    state,
    action,
    reward,
    next_state,
    time_step,
) -> TensorDict:
    return TensorDict(
        dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            time_step=time_step,
        ),
        # [NOTE] it is easier to think of everything as having a batch_dim.
    ).unsqueeze(0)
