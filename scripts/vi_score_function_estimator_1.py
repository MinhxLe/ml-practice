"""
example of variational inference with score function estimator.
Data is mixture of gaussian
"""

import attrs
from torch.utils.data import DataLoader
from mle.config import BaseCfg
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


@attrs.frozen
class Cfg(BaseCfg):
    project_name: str = "vi_with_score_function_estimator"
    dataset_size: int = 2000

    # training
    lr = 0.005
    n_epoch: int = 200
    batch_size: int = 100
    n_latent_samples: int = 20

    # logging
    eval_interval: int = 20


cfg = Cfg()


def generate_dataset(n: int) -> torch.Tensor:
    # gaussian mixture model
    mixture_weights = [0.2, 0.5, 0.3]
    means = [1, 5, 10]
    scales = [1, 1, 1]

    components = np.random.choice(len(mixture_weights), size=n, p=mixture_weights)
    data = np.stack([np.random.normal(means[i], scales[i]) for i in components])
    return torch.Tensor(data).view(n, 1)


class Model(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_components: int):
        super().__init__()
        self.mixing_logits = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
        )
        self.mu = nn.Parameter(torch.randn(n_components, n_features))
        # TODO assuming diagonal for now
        self.log_var = nn.Parameter(torch.ones((n_components, n_features)))

    def sample_latent(self, logits):
        distribution = torch.distributions.Categorical(logits=logits)
        samples = distribution.sample().to(Cfg.device)
        return samples, distribution.log_prob(samples)

    def forward(self, x):
        logits = self.mixing_logits(x)
        latent, latent_log_p = self.sample_latent(logits)
        return self.mu[latent], self.log_var[latent], latent_log_p


def loss_fn(x, mu, log_var, latent_log_p, n_categories):
    x = x.unsqueeze(-1)  # 1, batch, feature
    n_latent_samples, n_features = mu.shape
    var = torch.einsum(
        "bd,de->bde", log_var.exp(), torch.eye(n_features).to(cfg.device)
    )  # batch, feature, feature
    likelihood = (
        torch.distributions.MultivariateNormal(mu, var).log_prob(x).to(cfg.device)
    )
    # unfiform categorical
    prior_log_p = -torch.log(torch.tensor(n_categories, device=cfg.device))
    # p(x|z) - (q(z|x) - p(z))
    reward = (likelihood - (latent_log_p - prior_log_p)).sum(axis=0)
    return -(latent_log_p * reward.detach()).mean()


def eval_model(model, x):
    model.eval()
    with torch.no_grad():
        logits = model.mixing_logits(x).mean(axis=0)
        estimated_component = nn.Softmax()(logits)
        print(f"Estimated component weights: {estimated_component}")


np.random.seed(cfg.seed)
torch.random.manual_seed(cfg.seed)
x = generate_dataset(cfg.dataset_size).to(cfg.device)
dataloader = DataLoader(x, batch_size=cfg.batch_size)


model = Model(1, 4, 3).to(cfg.device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)


model.train()
for epoch in tqdm(range(cfg.n_epoch), total=cfg.n_epoch):
    total_loss = 0
    for _, x_batch in enumerate(dataloader):
        optimizer.zero_grad()
        mu, log_var, latent_log_p = model(x_batch)
        loss = loss_fn(x_batch, mu, log_var, latent_log_p, 3)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    if (epoch % cfg.eval_interval) == 0:
        eval_model(model, x)
