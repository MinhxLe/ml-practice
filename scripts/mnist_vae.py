"""
VAE reimplementation


TODO
[x] mnist dataset
[] cifar100 dataset

[x] vae implementation with reparametrization trick
[] training
[] log/wandb


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


class Cfg(BaseCfg):
    batch_size = 100

    # training
    n_epoch = 50
    lr = 1e-3

    test_loss_per_n_steps = 10


# data
train_dataset = datasets.MNIST(
    root=f"{Cfg.root_tmp_dir}/data/mnist_data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
test_dataset = datasets.MNIST(
    root=f"{Cfg.root_tmp_dir}/data/mnist_data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
train_dl = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=Cfg.batch_size, shuffle=True
)
test_dl = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=Cfg.batch_size, shuffle=True
)


model = VaeModel(
    x_dim=784,
    z_dim=2,
    encoder_hidden_dims=[512, 256],
    decoder_hidden_dims=[256, 512],
).to(Cfg.device)
optimizer = optim.Adam(model.parameters())

model.train()
for epoch_idx in tqdm(range(Cfg.n_epoch)):
    train_loss = 0
    for batch_idx, (x_batch, _) in enumerate(train_dl):
        optimizer.zero_grad()
        x_batch = x_batch.reshape((-1, 784)).to(Cfg.device)
        x_pred, mu, log_var = model(x_batch)
        loss = loss_function(x_batch, x_pred, mu, log_var)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # TODO add batch logging
    logger.info(f"Epoch {epoch_idx} Average loss: {train_loss/len(train_dataset):.4f}")
    # TODO add test logging
