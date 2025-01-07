"""
example of variational inference with score function estimator.
"""

from dataclasses import dataclass
from mle.config import BaseCfg
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Cfg(BaseCfg):
    project_name: str = "vi_with_score_function_estimator"
    n_samples: int = 1000


def generate_dataset(n: int):
    # gaussian mixture model
    mixture_weights = [0.2, 0.5, 0.3]
    means = [2, 5, 1]
    scales = [1, 1, 1]

    categories = np.random.choice(len(mixture_weights), size=n, p=mixture_weights)
    data = np.stack([np.random.normal(means[i], scales[i]) for i in categories])
    return data


cfg = Cfg()
np.random.seed(cfg.seed)
dataset = generate_dataset(cfg.n_samples)


class Model(nn.Module):
    def __init__(self, hidden_dim: int):
        pass

    def sample(self, logits):
        distribution = torch.distributions.Categorical(logits)
        pass

    def forward(self, x):
        pass
