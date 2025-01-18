import numpy as np
from mle.rl.core import Trajectory


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.all_metrics = []
        self.all_rewards = []

    def capture(self, trajs: list[Trajectory]):
        rewards = [t.total_reward() for t in trajs]
        metrics = dict(
            mean_traj_reward=np.mean(rewards),
            max_traj_reward=np.max(rewards),
            min_traj_reward=np.min(rewards),
            std_traj_reward=np.std(rewards),
        )
        self.all_rewards.append(rewards)
        self.all_metrics.append(metrics)
        return metrics
