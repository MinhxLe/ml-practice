from torch import nn


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
    for _ in range(n_hidden_layers):
        model.append(nn.Linear(hidden_dim, hidden_dim))
        model.append(activation_fn())
    model.append(nn.Linear(hidden_dim, output_dim))

    return model
