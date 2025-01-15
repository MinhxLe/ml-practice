from mle.utils.model_utils import build_simple_mlp


def test_build_simple_mlp():
    model = build_simple_mlp(1, 1, 1, 1)
    print(model)
