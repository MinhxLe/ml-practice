from typing import Callable, ClassVar
from torch import optim, nn


class BaseTrainer:
    loss_fn: ClassVar[Callable]

    def __init__(self, model: nn.Module, optimizer_cls, optim_kwargs: dict):
        self.model = model
        self.optimizer = optimizer_cls(model.parameters(), **optim_kwargs)

    def update(self, **kwargs) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(**kwargs)
        loss.backward()
        self.optimizer.step()
        return loss.item()
