"""
VAE reimplementation


TODO
[] mnist dataset
[] cifar100 dataset

[] vae implementation with reparametrization trick
[] training


[] evaluation
[] reconstruction exp
"""

import torch
from torch import nn


class Cfg:
    pass


class Encoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        pass
        # TODO return latent sample, mean, and log_variance


class Decoder(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )


class Vae(nn.Module):
    def __init__(self):
        super().__init__()
