import numpy as np
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, **kwargs):
        """
        Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            **kwargs:
        """

        self.state_size = state_size
        self.action_size = action_size
        print(f"Running {self.__class__.__name__} .... !!!!")

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        pass

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state: (array_like): current state
            eps: (float): epsilon, for epsilon-greedy action selection

        Returns:

        """

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor

        Returns:

        """
        raise NotImplementedError

    @staticmethod
    def compute_returns(rewards, next_value=0.0, gamma=0.99, dones=None):
        """
        Given a sequence of returns compute the vector of discounted returns
        [G_0, G_1, ..., G_T].
        Note that we alse use the trick of 'normalizing' the returns i.e.
        we subtract the mean and divide by the standard deviation.

        Args:
            rewards:
            next_value:
            gamma:
            dones:

        Returns:

        """
        if dones is not None:
            masks = 1 - dones
        else:
            masks = np.ones_like(rewards)
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)

        returns = np.array(returns)
        returns -= returns.mean()
        returns /= returns.std() if np.std(returns) > 0 else 1
        return returns

