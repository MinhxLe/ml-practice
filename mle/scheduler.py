import math


class ExponentialDecayScheduler:
    """
    A class that manages a value that decays exponentially from a start value to an end value.
    Commonly used for epsilon-greedy exploration in reinforcement learning.
    """

    def __init__(self, start_val: float, end_val: float, decay: float):
        """
        Initialize the decay scheduler.

        Args:
            start_val (float): Initial value
            end_val (float): Final value that it decays towards
            decay_rate (float): Decay rate - higher values mean slower decay
        """
        self.start_val = start_val
        self.end_val = end_val
        self.decay = decay
        self.steps_done = 0

    def step(self):
        """Increment the step counter"""
        self.steps_done += 1

    @property
    def value(self) -> float:
        return self.end_val + (self.start_val - self.end_val) * math.exp(
            -1.0 * self.steps_done * self.decay
        )
