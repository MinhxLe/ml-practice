from mle.rl.models.policy import CategoricalPolicy


def test_init_discrete_policy():
    print(CategoricalPolicy(2, 8, 8, 2))
