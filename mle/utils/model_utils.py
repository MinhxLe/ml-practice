from typing import TypeVar
import copy
from torch import nn


ModelT = TypeVar("ModelT", bound=nn.Module)


def build_simple_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    activation_fn=nn.ReLU,
) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation_fn(),
    )
    for _ in range(n_hidden_layers - 1):
        model.append(nn.Linear(hidden_dim, hidden_dim))
        model.append(activation_fn())
    model.append(nn.Linear(hidden_dim, output_dim))

    return model


def copy_model(model: ModelT) -> ModelT:
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(model.state_dict())
    return new_model


def polyak_update(model, target_model, factor):
    model_dict = model.state_dict()
    target_model_dict = target_model.state_dict()
    for key in model_dict:
        target_model_dict[key] = model_dict[key] * factor + target_model_dict[key] * (
            1 - factor
        )
    target_model.load_state_dict(target_model_dict)
