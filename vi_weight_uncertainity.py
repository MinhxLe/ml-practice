"""
implementation of https://arxiv.org/pdf/1505.05424
"""

import torch
from torch import nn, distributions
from dataclasses import dataclass


class LinearVariational(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        n_batches,
        bias=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self.n_batches = n_batches

        self.w_mu = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(0, 0.001)
        )

        # log(1+exp(p)) guarantees variance is >= 0
        self.w_p = nn.Parameter(
            torch.FloatTensor(in_features, out_features).normal_(-2.5, 0.001)
        )
        if self.include_bias:
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            # proxy for variance
            self.b_p = nn.Parameter(torch.zeros(out_features))

    def reparameterize(self, mu, p):
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn(mu.size())
        return mu + eps * sigma

    def kl(self, z, mu, p):
        sigma = torch.log(1 + torch.exp(p))
        # prior is standard normal distribution
        log_prior = distributions.Normal(0, 1).log_prob(z)
        log_q = distributions.Normal(mu, sigma).log_prob(z)
        # this can be mean if you mean likelihood loss
        return (log_q - log_prior).sum() * (1 / self.n_batches)

    def forward(self, x):
        W = self.reparameterize(self.w_mu, self.w_p)
        if self.include_bias:
            b = self.reparameterize(self.b_mu, self.b_p)
        else:
            b = 0

        z = x @ W + b

        kl_div = self.kl(W, self.w_mu, self.w_p)
        if self.include_bias:
            kl_div += self.kl(b, self.b_mu, self.b_p)

        return z, kl_div


class Model(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_batches):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LinearVariational(in_size, hidden_size, n_batches),
                nn.ReLU(),
                LinearVariational(hidden_size, hidden_size, n_batches),
                nn.ReLU(),
                LinearVariational(hidden_size, out_size, n_batches),
            ]
        )

    def forward(self, x):
        out = x
        accumulated_kl_div = 0
        for layer in self.layers:
            if isinstance(layer, LinearVariational):
                out, kl_div = layer(out)
                accumulated_kl_div += kl_div
            else:
                out = layer(out)
        return out, accumulated_kl_div
