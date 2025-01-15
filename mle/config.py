import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class BaseCfg:
    project_name: str
    root_tmp_dir: str = "./tmp"
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_wandb: bool = True

    @property
    def project_root_dir(self) -> str:
        return f"{self.root_tmp_dir}/{self.project_name}"
