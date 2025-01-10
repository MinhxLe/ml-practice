from collections import deque
import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.transitions = None

    def push(self, transition: TensorDict) -> None:
        # if transition.shape != torch.Size([]):
        #     raise ValueError("batch push not supported")

        if self.transitions is None:
            self.transitions = transition
        else:
            if self.transitions.shape[0] == self.max_size:
                self.transitions = self.transitions[1:]

            self.transitions = torch.cat([self.transitions, transition], dim=0)

    def sample(self, n: int = 1) -> TensorDict:
        if self.transitions is None:
            raise ValueError("no transitions to sample")
        curr_size = self.transitions.shape[0]
        idxs = torch.randperm(curr_size)[:n]
        return self.transitions[idxs]
