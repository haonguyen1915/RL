from rl.net.net import ReinforceBaseline
import numpy as np
import torch
import os
from torch import optim
from rl.agent.agent import Agent
from rl.memory.on_policy_replay import OnPolicyReplay

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 1  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
USE_CUDA = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def n(arr):
    return np.array(arr)


def t(arr):
    return torch.tensor(arr)


class ReinforceBaselineAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, **kwargs):
        """
        Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        super(ReinforceBaselineAgent, self).__init__(state_size, action_size, **kwargs)
        # Replay memory
        self.memory = OnPolicyReplay(BUFFER_SIZE)
        self.network = ReinforceBaseline(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.b_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.train_step(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:

        """
        state = torch.from_numpy(state).float().to(device)
        dist, value = self.network(state)
        # sample an action from the distribution
        action = dist.sample()
        return action.numpy()

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma: discount factor

        Returns:

        """
        # states, actions, rewards, next_states, dones = experiences
        states = experiences['states']
        for eps in range(len(states)):
            states = experiences['states'][eps]
            rewards = experiences['rewards'][eps]
            actions = experiences['actions'][eps]
            dones = experiences['dones'][eps]

            # Compute Gt or returns
            returns = self.compute_returns(rewards, 0.0, gamma, dones)
            returns_tensor = torch.tensor(returns)

            # Compute the loss for gradient descent
            states = torch.tensor(states).float()
            actions = torch.tensor(actions).long()

            dist, values = self.network(states)
            log_probs = dist.log_prob(actions)

            # compute the difference between the returns and the values
            delta = returns_tensor - values

            # compute the policy loss term. multiply the log probabilities by delta and sum
            # (remeber to call .detach() on delta since we do not want the gradient to propogate
            # to the value function network here)
            policy_loss = -torch.sum(log_probs * delta.detach())

            # compute the value function loss term
            value_function_loss = 0.5 * torch.sum(delta ** 2)

            # compute the composite loss
            loss = policy_loss + value_function_loss

            # update the policy and value function parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self, model_path):
        if not os.path.dirname(model_path):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.network.state_dict(), model_path)

    def load(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"File not found: {model_path}")
        self.network.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')))