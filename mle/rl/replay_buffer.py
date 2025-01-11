import torch
from tensordict import TensorDict


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.transitions = None

    def __len__(self):
        if self.transitions is None:
            return 0
        return self.transitions.shape[0]

    def push(self, transition: TensorDict) -> None:
        if transition.shape == torch.Size([]):
            transition = transition.unsqueeze(0)

        if self.transitions is None:
            self.transitions = transition
        else:
            if self.transitions.shape[0] == self.max_size:
                self.transitions = self.transitions[1:]
            try:
                self.transitions = torch.cat([self.transitions, transition], dim=0)
            except Exception:
                __import__("ipdb").set_trace()

    def sample(self, n: int = 1) -> TensorDict:
        if self.transitions is None:
            raise ValueError("no transitions to sample")
        curr_size = self.transitions.shape[0]
        idxs = torch.randperm(curr_size)[:n]
        return self.transitions[idxs]
