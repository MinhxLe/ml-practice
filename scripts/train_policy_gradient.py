from dataclasses import dataclass, asdict, field
from enum import StrEnum
import wandb
from mle.config import BaseCfg
from mle.rl.algo.policy_gradient import PolicyGradient, PolicyGradientCfg
import gymnasium as gym
from mle.rl.environment import GymEnv
from mle.rl.models.policy import CategoricalPolicy, GaussianPolicy
from mle.utils.project_utils import init_project


class EnvType(StrEnum):
    CARTPOLE = "cartpole"
    PENDULUM = "pendulum"


ENV_TYPE = EnvType.CARTPOLE


@dataclass(frozen=True)
class PolicyCfg:
    hidden_dim: int
    n_hidden_layers: int


@dataclass(frozen=True)
class Cfg(BaseCfg):
    policy_gradient_cfg: PolicyGradientCfg = field(default=NotImplemented)
    policy_cfg: PolicyCfg = PolicyCfg(
        hidden_dim=64,
        n_hidden_layers=1,
    )


project_name = f"pg_{ENV_TYPE}"
if ENV_TYPE == EnvType.CARTPOLE:
    env = GymEnv(gym.make("CartPole-v1"))
    cfg = Cfg(
        project_name=project_name,
        policy_gradient_cfg=PolicyGradientCfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=2000,
            max_episode_steps=200,
        ),
    )
elif ENV_TYPE == EnvType.PENDULUM:
    env = GymEnv(gym.make("InvertedPendulum-v4"))
    cfg = Cfg(
        project_name=project_name,
        policy_gradient_cfg=PolicyGradientCfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=10_000,
            max_episode_steps=1_000,
        ),
    )
else:
    raise NotImplementedError
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


if cfg.log_wandb:
    wandb.init(
        project=cfg.project_name,
        config=asdict(cfg),
    )
pg = PolicyGradient(create_policy, env, cfg.policy_gradient_cfg)
pg.train()
if cfg.log_wandb:
    wandb.finish()
