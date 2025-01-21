"""
script to train GAEestimator for different lambdas
"""

import attrs
from mle.rl.algo.gae import VPGWithGAE, VPGWithGAECfg
from mle.rl.env import GymEnv
import gymnasium as gym
from mle.rl.models.policy import CategoricalPolicy
from mle.utils import model_utils
from mle.utils.project_utils import init_project
from scripts.train_policy_gradient import Cfg, MODEL_CFG_MAP, EnvType
import wandb

ENV_TYPE = EnvType.CARTPOLE

env = GymEnv(gym.make("CartPole-v1"))
for lambda_ in [0, 0.2, 0.5, 0.9, 1]:
    cfg = Cfg(
        log_wandb=True,
        project_name="gae_cartpole",
        run_name=f"lambda_{lambda_}",
        policy_cfg=MODEL_CFG_MAP[EnvType.CARTPOLE],
        baseline_cfg=MODEL_CFG_MAP[EnvType.CARTPOLE],
        algo_cfg=VPGWithGAECfg(
            gamma=1.0,
            lr=3e-2,
            n_epochs=100,
            batch_size=2000,
            max_episode_steps=200,
            lambda_=lambda_,
        ),
    )
    init_project(cfg)
    policy = CategoricalPolicy(
        state_dim=env.state_dim,
        hidden_dim=cfg.policy_cfg.hidden_dim,
        n_actions=env.action_dim,
        n_hidden_layers=cfg.policy_cfg.n_hidden_layers,
    )
    baseline = model_utils.build_simple_mlp(
        input_dim=env.state_dim,
        hidden_dim=cfg.baseline_cfg.hidden_dim,
        n_hidden_layers=cfg.baseline_cfg.n_hidden_layers,
        output_dim=1,
    )
    pg = VPGWithGAE(
        policy=policy,
        baseline=baseline,
        env=env,
        cfg=cfg.algo_cfg,
    )
    if cfg.log_wandb:
        wandb.init(
            project=cfg.project_name,
            config=attrs.asdict(cfg),
            name=cfg.run_name,
        )
    try:
        pg.train()
    finally:
        if cfg.log_wandb:
            wandb.finish()
