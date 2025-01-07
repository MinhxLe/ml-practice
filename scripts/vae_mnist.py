"""
VAE reimplementation


TODO
[x] mnist dataset
[] cifar100 dataset

[x] vae implementation with reparametrization trick
[x] training
[x] log
[] add batch logging


[] evaluation
[] reconstruction exp
"""

import torch
from mle.config import BaseCfg
from mle.models.vae import VaeModel, loss_function
from torchvision import datasets, transforms
from tqdm import tqdm
from torch import optim
from loguru import logger
from torchvision.utils import save_image
from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg(BaseCfg):
    project_name: str = "vae_mnist"
    batch_size: int = 100

    # hyperparameters
    z_dim: int = 2

    # training
    n_epoch: int = 50
    lr: float = 1e-3

    # debug
    test_loss_per_n_steps: int = 10


cfg = Cfg()


# data
train_dataset = datasets.MNIST(
    root=f"{cfg.root_tmp_dir}/data/mnist_data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
test_dataset = datasets.MNIST(
    root=f"{cfg.root_tmp_dir}/data/mnist_data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
train_dl = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True
)
test_dl = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=cfg.batch_size, shuffle=True
)


model = VaeModel(
    x_dim=784,
    z_dim=cfg.z_dim,
    encoder_hidden_dims=[512, 256],
    decoder_hidden_dims=[256, 512],
).to(cfg.device)
optimizer = optim.Adam(model.parameters())

model.train()
for epoch_idx in tqdm(range(cfg.n_epoch)):
    train_loss = 0
    for batch_idx, (x_batch, _) in enumerate(train_dl):
        optimizer.zero_grad()
        x_batch = x_batch.view((-1, 784)).to(cfg.device)
        x_pred, mu, log_var = model(x_batch)
        loss = loss_function(x_batch, x_pred, mu, log_var)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # TODO add batch logging
    logger.info(f"Epoch {epoch_idx} Average loss: {train_loss/len(train_dataset):.4f}")
    # TODO add test logging

torch.save(model, f"{cfg.project_root_dir}/model")
model.eval()

# Experiment 1: Sample from latent space and using decoder
with torch.no_grad():
    z = torch.randn(64, cfg.z_dim).to(cfg.device)
    sample = model.decoder(z)
    save_image(
        sample.view(64, 1, 28, 28),
        f"{cfg.project_root_dir}/samples/sample_random.png",
    )


# Experiment 2: linear space
with torch.no_grad():
    # write me a linear sacp
    z = torch.linspace(-1, 1, 8)
    z = torch.cartesian_prod(z, z).to(cfg.device)
    sample = model.decoder(z)
    save_image(
        sample.view(64, 1, 28, 28),
        f"{cfg.project_root_dir}/samples/sample_linspace.png",
    )


# experiment 3: sample from posterior [not working as expected]
def sample_from_aggregated_posterior(dl):
    with torch.no_grad():
        aggr_z = None
        for batch_idx, (x_batch, _) in enumerate(dl):
            x_batch = x_batch.view((-1, 784)).to(cfg.device)
            z, _, _ = model.encoder(x_batch)
            if aggr_z is None:
                aggr_z = z
            else:
                aggr_z += z
        assert aggr_z is not None
        mean_z = aggr_z.sum(axis=0, keepdims=True) / len(dl)
        return model.decoder(mean_z)


sample = torch.stack(
    [sample_from_aggregated_posterior(train_dataset) for _ in range(64)]
)
save_image(
    sample.view(64, 1, 28, 28),
    f"{cfg.project_root_dir}/samples/sample_aggregated_posterior.png",
)


# Experiment 4: likelihood estimation from prior
# this does not work as well as we would like since the model is not
# fully probablistic.
# (see https://bjlkeng.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/)
def estimate_likelihood_from_prior(x, model: VaeModel, n: int):
    z = torch.randn((n, cfg.z_dim))
    distribution = torch.distributions.MultivariateNormal(
        torch.zeros(cfg.z_dim), torch.eye(cfg.z_dim)
    )
    distribution.log_prob(z)

    pass
