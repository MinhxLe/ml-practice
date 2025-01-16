from typing import Callable
from torch import nn


class BaseTrainer:
    loss_fn: Callable
    update: Callable[..., float]

    def __init__(self, model: nn.Module, optimizer_cls, optim_kwargs: dict):
        self.model = model
        self.optimizer = optimizer_cls(model.parameters(), **optim_kwargs)

    def _step_optimizer(self, **loss_kwargs) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.loss_fn(**loss_kwargs)
        loss.backward()
        self.optimizer.step()
        return loss.item()
