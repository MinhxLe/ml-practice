"""
simple implementation fo VI with score function estimttor.
Data is bernoulli
"""

from dataclasses import dataclass

from torch.utils.data import DataLoader
from mle.config import BaseCfg
import torch
from torch import nn


@dataclass(frozen=True)
class Cfg(BaseCfg):
    project_name: str = "vi_with_score_function_estimator"
    n_samples: int = 1000

    # prior
    alpha = 1.0
    beta = 1.0

    # training
    lr = 0.005
    n_epoch: int = 2_000
    batch_size: int = 1000
    n_latent_samples: int = 20

    eval_interval: int = 100


cfg = Cfg()
torch.set_default_device(cfg.device)


def generate_dataset(n: int) -> torch.Tensor:
    p = torch.tensor(0.7)
    return torch.distributions.Bernoulli(probs=p).sample((n,))


class Model(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self):
        # exp() to be nonnegative
        return torch.distributions.Beta(self.alpha.exp(), self.beta.exp())


def loss_fn(x, p, log_q):
    x = x.unsqueeze(-1)
    log_likelihood = torch.distributions.Bernoulli(p).log_prob(x)
    prior_log_p = torch.distributions.Beta(cfg.alpha, cfg.beta).log_prob(p)
    elbo = (log_likelihood - (log_q - prior_log_p)).sum(axis=0)
    return -(log_q * (elbo.detach())).mean()


def evaluate_model(model, true_p=0.7):
    model.eval()
    with torch.no_grad():
        # Get variational parameters
        alpha = model.alpha.item()
        beta = model.beta.item()

        # Compute mean of Beta distribution
        estimated_p = alpha / (alpha + beta)

        # Sample from posterior
        samples = torch.distributions.Beta(alpha, beta).sample((1000,))

        print(f"True p: {true_p:.3f}")
        print(f"Estimated p: {estimated_p:.3f}")
        print(f"Alpha: {alpha:.3f}")
        print(f"Beta: {beta:.3f}")

        return samples


torch.random.manual_seed(cfg.seed)
x = generate_dataset(cfg.n_samples)
dataloader = DataLoader(x, batch_size=cfg.batch_size)

model = Model(cfg.alpha, cfg.beta).to(device=cfg.device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

model.train()
for epoch in range(cfg.n_epoch):
    for _, x_batch in enumerate(dataloader):
        optimizer.zero_grad()
        distribution = model()

        p = distribution.sample((cfg.n_latent_samples,))
        log_q = distribution.log_prob(p)
        loss = loss_fn(x_batch, p, log_q)
        loss.backward()
        optimizer.step()
    if (epoch % cfg.eval_interval) == 0:
        evaluate_model(model)
