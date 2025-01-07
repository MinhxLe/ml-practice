"""
source: https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/
"""

from typing import Tuple
import numpy as np
import torch
from torch import nn, distributions
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


class Config:
    seed = 42
    w0 = 0.125
    b0 = 5.0
    x_range = [-20, 60]
    n_train = 150
    n_test = 50
    batch_size = 50
    lr = 0.01
    n_epoch = 400
    # prior
    mu0 = 0
    log_var0 = torch.log(torch.tensor(1.0))


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def generate_dataset(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x_range = Config.x_range
    w0 = Config.w0
    b0 = Config.b0

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2)

    # uniform
    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]

    # gaussian
    eps = np.random.randn(n) * s(x)

    y = (w0 * x * (1 + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()  # normalize

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return y[:, None], x[:, None]


def plot_graph(x, y) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.7, label="Generated Data")
    plt.title("Generated Dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def eval_model(
    model: nn.Module,
    loss_fn,
    dataset: Dataset,
):
    model.eval()
    total_loss = 0
    total_count = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=Config.batch_size)
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(dataloader):
            total_loss += loss_fn(y_batch, model(x_batch))
            total_count += len(x_batch)
    return total_loss / total_count


def train_model(
    model: nn.Module,
    loss_fn,
    train_dataset: GeneratedDataset,
    test_dataset: GeneratedDataset | None,
):
    optim = torch.optim.Adam(model.parameters(), lr=Config.lr)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True
    )
    for epoch in tqdm(range(Config.n_epoch)):
        if test_dataset is not None:
            eval_loss = eval_model(model, loss_fn, test_dataset)
            print(f"Epoch: {epoch}/{Config.n_epoch}; eval loss: {eval_loss}")
        model.train()
        for _, (x_batch, y_batch) in enumerate(train_dataloader):
            optim.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)
            loss.backward()
            optim.step()


np.random.seed(Config.seed)
y, x = generate_dataset(Config.n_train + Config.n_test)
y_tensor = torch.tensor(y, dtype=torch.float)
x_tensor = torch.tensor(x, dtype=torch.float)
dataset = GeneratedDataset(x, y)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [Config.n_train, Config.n_test],
)
# plot_graph(x, y)


# MLE Modelling
class NNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Sequential(nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 1))

    def forward(self, x):
        return self.out(x)


model = NNModel()


def loss_fn(y_pred, y_actual):
    # mse loss
    return (0.5 * (y_pred - y_actual) ** 2).mean()


train_model(model, loss_fn, train_dataset, test_dataset)


model.eval()
with torch.no_grad():
    y_pred = model(x_tensor)


plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label="Generated Data")
plt.scatter(x, y_pred, alpha=0.7, label="mle prediction")
plt.title("Generated Dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()


# VI Posterior Modelling
class VIModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)

        # reparametrization trick so we can backprop through mu and log_var
        std = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn(mu.size())
        y_posterior = mu + eps * std
        return y_posterior, mu, log_var


# loss
def log_gaussian(y, mu, log_var):
    var = torch.exp(log_var)
    return -0.5 * torch.log(2 * np.pi * var) - (1 / (2 * var)) * (y - mu) ** 2


def elbo(y_pred, y_true, mu, log_var):
    # elbo is E_{z~q}[log(p(x=D,z)/q(z)]
    # E_{z~q}[p(x=D|z)] - KL(q||p(z)
    # is the latent y_pred the mean?
    log_likelihood = log_gaussian(y_true, mu, log_var)
    # log_likelihood = log_gaussian(y_true, y_pred, log_var)
    log_prior = log_gaussian(
        y_pred,
        mu=Config.mu0,
        log_var=Config.log_var0,
    )
    # variational propability E[log(q(z)]
    log_p_q = log_gaussian(y_pred, mu, log_var)
    return (log_likelihood + log_prior - log_p_q).mean()


def vi_loss(y_true, model_output):
    y_pred, mu, log_var = model_output
    return -elbo(y_pred, y_true, mu, log_var)


vi_model = VIModel()
train_model(vi_model, vi_loss, train_dataset, test_dataset)

with torch.no_grad():
    y_pred = torch.cat([vi_model(x_tensor)[0] for _ in range(1000)], dim=1)

# Get some quantiles
q1, mu, q2 = np.quantile(y_pred, [0.05, 0.5, 0.95], axis=1)

plt.figure(figsize=(16, 6))
plt.scatter(x_tensor, y_tensor)
plt.plot(x, mu)
plt.fill_between(x.flatten(), q1, q2, alpha=0.2)
plt.show()


# another implementation of loss is the closed form KL
# TODO why is this different from the previous implementation
def vi_loss2(y_true, model_output):
    y_pred, mu, log_var = model_output
    reconstruction_loss = (0.5 * (y_pred - y_true) ** 2).sum()
    kl_divergence = (0.5 * (torch.exp(log_var) + torch.exp(mu) - 1 - log_var)).sum()
    return reconstruction_loss + kl_divergence


vi_model2 = VIModel()
train_model(vi_model2, vi_loss2, train_dataset, test_dataset)

with torch.no_grad():
    y_pred = torch.cat([vi_model2(x_tensor)[0] for _ in range(1000)], dim=1)

# Get some quantiles
q1, mu, q2 = np.quantile(y_pred, [0.05, 0.5, 0.95], axis=1)

plt.figure(figsize=(16, 6))
plt.scatter(x_tensor, y_tensor)
plt.plot(x, mu)
plt.fill_between(x.flatten(), q1, q2, alpha=0.2)
plt.show()

#
from vi_weight_uncertainity import Model

n_batches = len(train_dataset) // Config.batch_size
model = Model(1, 20, 1, n_batches=n_batches)


def loss_fn(y_true, model_output):
    y_pred, kl_div = model_output
    reconstruction_loss = -distributions.Normal(y_pred, 0.0001).log_prob(y_true).sum()
    return reconstruction_loss + kl_div


train_model(model, loss_fn, train_dataset, None)

model.eval()
with torch.no_grad():
    # sample per x 1000 times
    trace = np.array([model(x_tensor)[0].flatten().numpy() for _ in range(1000)]).T
q_25, q_95 = np.quantile(trace, [0.05, 0.95], axis=1)
plt.figure(figsize=(16, 6))
plt.plot(x_tensor, trace.mean(axis=1))
plt.scatter(x_tensor, y_tensor)
plt.fill_between(x_tensor.flatten(), q_25, q_95, alpha=0.2)
plt.show()
