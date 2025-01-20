import numpy as np
import abc
import matplotlib.pyplot as plt
from typing import List, Tuple


class BernoulliBandit:
    def __init__(self, n_actions: int, seed: int | None = None):
        self.n_arms = n_actions
        if seed is not None:
            np.random.seed(seed)

        # Generate random probabilities for each arm
        self.probabilities = np.random.random(n_actions)

    def step(self, action: int) -> int:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid arm index: {action}")

        return np.random.binomial(1, self.probabilities[action])

    def regret(self, action):
        pass
        return np.max(self.probabilities) - self.probabilities[action]


class Strategy(abc.ABC):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @abc.abstractmethod
    def act(self) -> np.int64 | int:
        pass

    @abc.abstractmethod
    def update(self, action, reward):
        pass


class RandomStrategy(Strategy):
    def __str__(self) -> str:
        return "random"

    def act(self) -> np.int64:
        return np.random.choice(range(0, self.n_actions))

    def update(self, action, reward):
        return


class EpsilonGreedyStrategy(Strategy):
    def __init__(self, n_actions, eps):
        super().__init__(n_actions)
        self.eps = eps
        self.counts = np.zeros((n_actions,))
        self.rewards = np.zeros((n_actions,))

    def __str__(self) -> str:
        return f"eps_greedy_{self.eps}"

    def act(self):
        if np.random.rand() < self.eps:
            return np.random.choice(range(0, self.n_actions))
        else:
            return np.argmax(self.rewards / (self.counts + 1e-6))

    def update(self, action, reward):
        self.counts[action] += 1
        self.rewards[action] += reward


class UCBStrategy(Strategy):
    def __init__(self, n_actions, delta):
        super().__init__(n_actions)
        self.t = 0
        self.delta = delta
        self.counts = np.zeros((n_actions,))
        self.rewards = np.zeros((n_actions,))

    def __str__(self) -> str:
        return f"ucb_{self.delta}"

    def act(self):
        self.t += 1
        emperical_reward_mean = self.rewards / (self.counts + 1e-8)
        gap = np.sqrt(2 * np.log(self.t / self.delta) / (self.counts + 1e-8))
        return np.argmax(emperical_reward_mean + gap)

    def update(self, action, reward):
        self.counts[action] += 1
        self.rewards[action] += reward


def run_simulation(
    bandit: BernoulliBandit,
    strategy: Strategy,
    n_steps: int,
) -> Tuple[List[int], List[float], List[float]]:
    rewards = []
    cumulative_rewards = []
    cumulative_regrets = []
    cumulative_regret = 0
    cumulative_reward = 0

    for t in range(n_steps):
        action = strategy.act()
        reward = bandit.step(action)

        # Update strategy
        strategy.update(action, reward)

        # Store results
        rewards.append(reward)
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)

        cumulative_regret += bandit.regret(action)
        cumulative_regrets.append(cumulative_regret)

    return rewards, cumulative_rewards, cumulative_regrets


def plot_results(results: dict, n_steps: int):
    """
    Plot results from different strategies.

    Args:
        results: Dictionary mapping strategy names to cumulative rewards
        n_steps: Number of steps in simulation
    """
    plt.figure(figsize=(10, 6))

    for name, cumulative_rewards in results.items():
        plt.plot(range(n_steps), cumulative_rewards, label=name)

    plt.xlabel("Time steps")
    plt.ylabel("Cumulative reward")
    plt.title("Comparison of Bandit Strategies")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Parameters
    N_ACTIONS = 10
    N_STEPS = 100_000
    SEED = 42

    # Create environment
    bandit = BernoulliBandit(N_ACTIONS, SEED)
    all_cumulative_regrets = dict()
    strategies = [
        RandomStrategy(N_ACTIONS),
        EpsilonGreedyStrategy(N_ACTIONS, 0.1),
        EpsilonGreedyStrategy(N_ACTIONS, 0),
        UCBStrategy(N_ACTIONS, 0.01),
        UCBStrategy(N_ACTIONS, 0.001),
    ]
    for strategy in strategies:
        _, _, cumulative_regrets = run_simulation(bandit, strategy, N_STEPS)
        all_cumulative_regrets[str(strategy)] = cumulative_regrets

    plot_results(all_cumulative_regrets, N_STEPS)
