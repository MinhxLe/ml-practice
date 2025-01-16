import datetime
import attrs
from enum import StrEnum
import wandb
from mle.config import BaseCfg
from mle.rl.algo.vpg import (
    VPG,
    VPGCfg,
)
import gymnasium as gym
from mle.rl.algo.ppo import PPO, PPOCfg
from mle.rl.env import GymEnv
from mle.rl.models.policy import CategoricalPolicy, GaussianPolicy
from mle.utils import model_utils
from mle.utils.project_utils import init_project


class EnvType(StrEnum):
    CARTPOLE = "cartpole"
    PENDULUM = "pendulum"
    CHEETAH = "cheetah"


class AlgoType(StrEnum):
    VANILLA = "vanilla"
    PPO = "ppo"


SEED = 1
ENV_TYPE = EnvType.CHEETAH
ALGO_TYPE = AlgoType.VANILLA
PROJECT_NAME = f"{ENV_TYPE}"
RUN_NAME = f"{ALGO_TYPE}_{SEED}_{datetime.datetime.now()}"


@attrs.frozen
class ModelCfg:
    hidden_dim: int
    n_hidden_layers: int


@attrs.frozen
class Cfg(BaseCfg):
    algo_cfg: VPGCfg | PPOCfg
    run_name: str
    baseline_cfg: ModelCfg
    policy_cfg: ModelCfg
    seed: int = 1
    log_wandb: bool = True


MODEL_CFG_MAP = {
    EnvType.CARTPOLE: ModelCfg(
        hidden_dim=64,
        n_hidden_layers=1,
    ),
    EnvType.PENDULUM: ModelCfg(
        hidden_dim=64,
        n_hidden_layers=1,
    ),
    EnvType.CHEETAH: ModelCfg(
        hidden_dim=64,
        n_hidden_layers=2,
    ),
}

ALGO_CFG_MAP = {
    AlgoType.VANILLA: {
        EnvType.CARTPOLE: VPGCfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=2000,
            max_episode_steps=200,
        ),
        EnvType.PENDULUM: VPGCfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=10_000,
            max_episode_steps=1_000,
        ),
        EnvType.CHEETAH: VPGCfg(
            gamma=0.9,
            lr=3e-2,
            n_epochs=200,
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
        EnvType.PENDULUM: PPOCfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=10_000,
            max_episode_steps=1_000,
            clipped_eps=0.2,
            n_gradient_steps=20,
        ),
        EnvType.CHEETAH: PPOCfg(
            gamma=0.9,
            lr=1e-2,
            n_epochs=200,
            batch_size=10_000,
            max_episode_steps=1_000,
            clipped_eps=0.2,
            n_gradient_steps=10,
        ),
    },
}
ENV_NAME_MAP = {
    EnvType.CARTPOLE: "CartPole-v1",
    EnvType.PENDULUM: "InvertedPendulum-v4",
    EnvType.CHEETAH: "HalfCheetah-v4",
}
ALGO_CLS_MAP = {
    AlgoType.VANILLA: VPG,
    AlgoType.PPO: PPO,
}


env = GymEnv(gym.make(ENV_NAME_MAP[ENV_TYPE]))
cfg = Cfg(
    project_name=PROJECT_NAME,
    run_name=RUN_NAME,
    policy_cfg=MODEL_CFG_MAP[ENV_TYPE],
    baseline_cfg=MODEL_CFG_MAP[ENV_TYPE],
    algo_cfg=ALGO_CFG_MAP[ALGO_TYPE][ENV_TYPE],
)
init_project(cfg)


if env.is_discrete:
    policy = CategoricalPolicy(
        state_dim=env.state_dim,
        hidden_dim=cfg.policy_cfg.hidden_dim,
        n_actions=env.action_dim,
        n_hidden_layers=cfg.policy_cfg.n_hidden_layers,
    )
else:
    policy = GaussianPolicy(
        state_dim=env.state_dim,
        hidden_dim=cfg.policy_cfg.hidden_dim,
        action_dim=env.action_dim,
        n_hidden_layers=cfg.policy_cfg.n_hidden_layers,
    )
baseline = model_utils.build_simple_mlp(
    input_dim=env.state_dim,
    hidden_dim=cfg.baseline_cfg.hidden_dim,
    n_hidden_layers=cfg.baseline_cfg.n_hidden_layers,
    output_dim=1,
)

if cfg.log_wandb:
    wandb.init(
        project=cfg.project_name,
        config=attrs.asdict(cfg),
        name=cfg.run_name,
    )
pg = ALGO_CLS_MAP[ALGO_TYPE](
    policy=policy,
    baseline=baseline,
    env=env,
    cfg=cfg.algo_cfg,
)
try:
    pg.train()
finally:
    if cfg.log_wandb:
        wandb.finish()
