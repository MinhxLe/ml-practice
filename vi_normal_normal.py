"""
An example of VI over an exponential-normal model.
"""

from typing import Tuple
import torch
import numpy as np
import torch.distributions as torch_d

Z_RATE = 1.0
X_VAR = 1.0
N = 1_000


def sample_exponential(rate: float, n: int) -> torch.Tensor:
    distribution = torch_d.exponential.Exponential(rate)
    return distribution.sample((n,))


def exponential_normal_model(
    z_rate: float,
    x_var: float,
    n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_df = sample_exponential(z_rate, n)
    x_df = torch.normal(z_df, np.sqrt(x_var))
    return z_df, x_df


z_df, x_df = exponential_normal_model(Z_RATE, X_VAR, N)


# from x_df, we want to do inference about z_df but the true
# posterior is intractable. instead we use VI to find a surogate posterior
#
