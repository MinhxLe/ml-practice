import torch
from mle.rl.core import Trajectory


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_metrics = []
        self.all_rewards = []

    def capture(self, trajs: list[Trajectory]):
        traj_rewards = torch.stack([t.to_tensordict()["reward"].sum() for t in trajs])
        metrics = dict(
            mean_traj_reward=traj_rewards.mean(),
            # last_traj_reward=traj_rewards[-1],
            max_traj_reward=traj_rewards.max(),
        )
        self.all_rewards.append(traj_rewards.cpu().numpy())
        self.all_metrics.append(metrics)
        return metrics
