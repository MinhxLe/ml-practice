import torch


class BaseCfg:
    root_tmp_dir = "./tmp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
