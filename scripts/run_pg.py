from dataclasses import dataclass, asdict
import wandb
from mle.config import BaseCfg
from mle.rl.algo.policy_gradient import PolicyGradient, PolicyGradientCfg
import gymnasium as gym
from mle.rl.environment import GymEnv
from mle.rl.models.policy import DiscretePolicy
from mle.utils.project_utils import init_project


@dataclass(frozen=True)
class PolicyCfg:
    hidden_dim: int
    n_hidden_layers: int


@dataclass(frozen=True)
class Cfg(BaseCfg):
    project_name: str = "pg_cartpole_v2"
    policy_cfg: PolicyCfg = PolicyCfg(
        hidden_dim=64,
        n_hidden_layers=1,
    )
    policy_gradient_cfg: PolicyGradientCfg = PolicyGradientCfg(
        gamma=1.0,
        lr=4e-2,
        n_epochs=100,
        batch_size=2000,
        max_episode_steps=200,
        log_wandb=False,
        train_log_freq=1,
    )

    max_episode_steps = 200
    log_wandb: bool = False


cfg = Cfg(log_wandb=True)
env = GymEnv(gym.make("CartPole-v1"))
init_project(cfg)


def create_policy():
    if env.is_discrete:
        return DiscretePolicy(
            state_dim=env.state_dim,
            hidden_dim=cfg.policy_cfg.hidden_dim,
            n_actions=env.action_dim,
            n_hidden_layers=cfg.policy_cfg.n_hidden_layers,
        )
    raise NotImplementedError


if cfg.log_wandb:
    wandb.init(
        project=cfg.project_name,
        config=asdict(cfg),
    )
pg = PolicyGradient(create_policy, env, cfg.policy_gradient_cfg)
pg.train()
if cfg.log_wandb:
    wandb.finish()
