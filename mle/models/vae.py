import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(nn.ReLU())

        self.shared_layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dims[0]),
            nn.ReLU(),
            *hidden_layers,
        )
        self.mu = nn.Linear(hidden_dims[-1], z_dim)
        self.log_var = nn.Linear(hidden_dims[-1], z_dim)

    def sample(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu).to(mu.device)
        return (sigma * eps) + mu

    def forward(self, x):
        shared = self.shared_layers(x)  # batch, hidden
        mu = self.mu(shared)  # batch, z
        log_var = self.log_var(shared)  # batch,z

        z = self.sample(mu, log_var)
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        hidden_layers = []
        for i in range(len(hidden_dims) - 1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            hidden_layers.append(nn.ReLU())
        self.layers = nn.Sequential(
            nn.Linear(z_dim, hidden_dims[0]),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hidden_dims[-1], x_dim),
            # [TODO] activation can be specified outside of this module
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.layers(z)


class VaeModel(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        encoder_hidden_dims: list[int],
        decoder_hidden_dims: list[int],
    ):
        super().__init__()
        self.encoder = Encoder(
            x_dim=x_dim, z_dim=z_dim, hidden_dims=encoder_hidden_dims
        )
        self.decoder = Decoder(
            x_dim=x_dim, z_dim=z_dim, hidden_dims=decoder_hidden_dims
        )

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        return self.decoder(z), mu, log_var


def loss_function(x_true, x_pred, mu, log_var):
    # bse because regression of [0, 1]. this can be interpreted as P(x|z)
    bse_loss = F.binary_cross_entropy(x_pred, x_true, reduction="sum")

    # [TODO] fix this implementation
    var = torch.exp(log_var)
    trace = torch.sum(var)
    log_det = torch.sum(log_var)
    z_dim = mu.shape[0]
    kl = 0.5 * (trace + (mu**2).sum() - z_dim - log_det)

    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return torch.sum(bse_loss + kl)
