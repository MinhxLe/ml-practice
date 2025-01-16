from mle.rl.algo.policy_gradient import (
    BasePolicyGradient,
    BasePolicyGradientCfg,
    BasePolicyTrainer,
)
import torch
from mle.rl.core import Trajectory, calculate_returns


class VanillaPolicyTrainer(BasePolicyTrainer):
    def _calculate_advantages(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
    ):
        if self.baseline is not None:
            with torch.no_grad():
                advantages = returns - self.baseline(states).squeeze(1)
        else:
            advantages = returns

        mu = torch.mean(advantages)
        std = torch.std(advantages)
        return (advantages - mu) / std

    def loss_fn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ):
        action_log_probs = self.policy.action_dist(states).log_prob(actions)
        return -(action_log_probs * advantages).mean()

    def update(self, trajs: list[Trajectory]) -> float:
        traj_tds = [t.to_tensordict() for t in trajs]
        all_states = torch.concat([t["state"] for t in traj_tds])
        all_actions = torch.concat([t["action"] for t in traj_tds])
        all_returns = torch.concat([calculate_returns(t, self.gamma) for t in trajs])
        all_advantages = self._calculate_advantages(all_returns, all_states)
        self.optimizer.zero_grad()
        loss = self.loss_fn(
            states=all_states, advantages=all_advantages, actions=all_actions
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()


class VanillaPolicyGradientCfg(BasePolicyGradientCfg):
    pass


class VanillaPolicyGradient(BasePolicyGradient[VanillaPolicyGradientCfg]):
    def _init_policy_trainer(self):
        return VanillaPolicyTrainer(
            self.policy,
            lr=self.cfg.lr,
            baseline=self.baseline,
            gamma=self.cfg.gamma,
        )
