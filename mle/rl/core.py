from typing import List
from tensordict import TensorDict
import torch


class Transition(TensorDict):
    @property
    def state(self) -> torch.Tensor:
        return self["state"]

    @property
    def action(self) -> torch.Tensor:
        return self["action"]

    @property
    def reward(self) -> torch.Tensor:
        return self["reward"]

    @property
    def time_step(self) -> torch.Tensor:
        return self["time_step"]

    @property
    def terminated(self) -> torch.Tensor:
        return self["terminated"].to(torch.bool)

    @property
    def next_state(self) -> torch.Tensor:
        return self["next_state"]


# TODO figure out howt o add a batch dim here
Transitions = Transition


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
    )


class Trajectory(list[Transition]):
    def __init__(self, transitions: List[Transition] | None = None):
        if transitions is None:
            transitions = []
        transitions.sort(key=lambda s: s.time_step.item())
        assert all(t.batch_size == torch.Size([]) for t in transitions)
        super().__init__(transitions)

    @property
    def terminated(self) -> bool:
        if len(self) == 0:
            return False
        return self[-1].terminated.item()

    def to_tensordict(self) -> TensorDict:
        return torch.stack(self)

    def to_transitions(self) -> Transitions:
        return Transitions(torch.stack(self), batch_size=torch.Size([len(self)]))

    def total_reward(self) -> float:
        return self.to_transitions().reward.sum().item()


def calculate_returns(traj: Trajectory, gamma: float) -> torch.Tensor:
    returns = []
    current_return = 0.0
    for transition in reversed(traj):
        current_return = transition.reward + gamma * current_return
        returns.append(current_return)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)
