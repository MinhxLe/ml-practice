from dataclasses import dataclass, asdict
import wandb
from mle.config import BaseCfg
from mle.rl.algo.policy_gradient import PolicyGradient, PolicyGradientCfg
import gymnasium as gym
from mle.rl.environment import GymEnv
from mle.rl.models.policy import DiscretePolicy, GaussianPolicy
from mle.utils.project_utils import init_project


@dataclass(frozen=True)
class PolicyCfg:
    hidden_dim: int
    n_hidden_layers: int


@dataclass(frozen=True)
class Cfg(BaseCfg):
    project_name: str = "pg_pendulum"
    policy_cfg: PolicyCfg = PolicyCfg(
        hidden_dim=64,
        n_hidden_layers=1,
    )
    policy_gradient_cfg: PolicyGradientCfg = PolicyGradientCfg(
        gamma=1.0,
        lr=3e-2,
        n_epochs=100,
        batch_size=10_000,
        max_episode_steps=1_000,
        log_wandb=True,
        train_log_freq=1,
    )

    max_episode_steps = 200
    log_wandb: bool = True


cfg = Cfg(log_wandb=True)
env = GymEnv(gym.make("InvertedPendulum-v4"))
init_project(cfg)


def create_policy():
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
