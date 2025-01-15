import attrs
from enum import StrEnum
import wandb
from mle.config import BaseCfg
from mle.rl.algo.policy_gradient import PolicyGradient, PolicyGradientCfg
import gymnasium as gym
from mle.rl.algo.ppo import PPO, PPOCfg
from mle.rl.env import GymEnv
from mle.rl.models.policy import CategoricalPolicy, GaussianPolicy
from mle.utils import model_utils
from mle.utils.project_utils import init_project


class EnvType(StrEnum):
    CARTPOLE = "cartpole"
    PENDULUM = "pendulum"


class AlgoType(StrEnum):
    VANILLA = "vanilla"
    PPO = "ppo"


SEED = 1
ENV_TYPE = EnvType.CARTPOLE
ALGO_TYPE = AlgoType.PPO
PROJECT_NAME = f"{ENV_TYPE}"
RUN_NAME = f"{ALGO_TYPE}_{SEED}"


@attrs.frozen
class ModelCfg:
    hidden_dim: int
    n_hidden_layers: int


@attrs.frozen
class Cfg(BaseCfg):
    algo_cfg: PolicyGradientCfg | PPOCfg = attrs.field()
    run_name: str = attrs.field()
    seed: int = 1
    # baseline_cfg: ModelCfg | None = None
    baseline_cfg: ModelCfg | None = ModelCfg(
        hidden_dim=64,
        n_hidden_layers=1,
    )
    policy_cfg: ModelCfg = ModelCfg(
        hidden_dim=64,
        n_hidden_layers=1,
    )


ALGO_CFG_MAP = {
    AlgoType.VANILLA: {
        EnvType.CARTPOLE: PolicyGradientCfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=2000,
            max_episode_steps=200,
        ),
        EnvType.PENDULUM: PolicyGradientCfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=10_000,
            max_episode_steps=1_000,
        ),
    },
    AlgoType.PPO: {
        EnvType.CARTPOLE: PPOCfg(
            gamma=1.0,
            n_epochs=100,
            batch_size=2000,
            lr=3e-2,
            max_episode_steps=200,
            clipped_eps=0.2,
            n_gradient_steps=10,
        ),
    },
}
ENV_NAME_MAP = {
    EnvType.CARTPOLE: "CartPole-v1",
    ENV_TYPE.PENDULUM: "InvertedPendulum-v4",
}
ALGO_CLS_MAP = {
    AlgoType.VANILLA: PolicyGradient,
    AlgoType.PPO: PPO,
}


env = GymEnv(gym.make(ENV_NAME_MAP[ENV_TYPE]))
cfg = Cfg(
    project_name=PROJECT_NAME,
    run_name=RUN_NAME,
    algo_cfg=ALGO_CFG_MAP[ALGO_TYPE][ENV_TYPE],
)
init_project(cfg)


def create_policy():
    if env.is_discrete:
        return CategoricalPolicy(
            state_dim=env.state_dim,
            hidden_dim=cfg.policy_cfg.hidden_dim,
            n_actions=env.action_dim,
            n_hidden_layers=cfg.policy_cfg.n_hidden_layers,
        )
    else:
        return GaussianPolicy(
            state_dim=env.state_dim,
            hidden_dim=cfg.policy_cfg.hidden_dim,
            action_dim=env.action_dim,
            n_hidden_layers=cfg.policy_cfg.n_hidden_layers,
        )


if cfg.baseline_cfg is not None:

    def create_baseline():
        return model_utils.build_simple_mlp(
            input_dim=env.state_dim,
            hidden_dim=cfg.baseline_cfg.hidden_dim,
            n_hidden_layers=cfg.baseline_cfg.n_hidden_layers,
            output_dim=1,
        )
else:
    create_baseline = None

if cfg.log_wandb:
    wandb.init(
        project=cfg.project_name,
        config=attrs.asdict(cfg),
        name=cfg.run_name,
    )
pg = ALGO_CLS_MAP[ALGO_TYPE](
    create_policy_fn=create_policy,
    create_baseline_fn=create_baseline,
    env=env,
    cfg=cfg.algo_cfg,
)
pg.train()
if cfg.log_wandb:
    wandb.finish()
