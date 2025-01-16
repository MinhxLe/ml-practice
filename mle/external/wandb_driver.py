import wandb


def is_initialized():
    return hasattr(wandb, "run") and wandb.run is not None
