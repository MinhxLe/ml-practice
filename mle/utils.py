from mle.config import BaseCfg
import torch
import matplotlib


def init_project(cfg: BaseCfg):
    torch.manual_seed(cfg.seed)
    torch.set_default_device(cfg.device)
    matplotlib.use("qtagg")
