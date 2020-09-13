from rl.net.net import Reinforce, ActorCritic
from rl.memory.replay import Replay
import numpy as np
import torch
import os
from torch import optim
from rl.agent.agent import Agent
from rl.memory.on_policy_replay import OnPolicyReplay

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 100  # minibatch size
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


class A2CAgent(Agent):
    def __init__(self, state_size, action_size, **kwargs):
        super(A2CAgent, self).__init__(state_size, action_size, **kwargs)
        self.network = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.memory = Replay(BUFFER_SIZE, BATCH_SIZE)

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

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample(clear=True)
            self.train_step(experiences, GAMMA)

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma: discount factor

        Returns:

        """
        states = experiences['states']
        rewards = experiences['rewards']
        actions = experiences['actions']
        next_states = experiences['next_states']
        dones = experiences['dones']

        # Compute the next value
        last_state = torch.tensor(next_states[-1]).float()
        _, next_value = self.network(last_state)

        # Calculate the discounted return of the episode
        returns = self.compute_returns(rewards, next_value.item(), gamma, dones)
        returns_tensor = torch.tensor(returns)

        # Compute the loss for gradient descent
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).long()

        dist, values = self.network(states)
        log_probs = dist.log_prob(actions)

        # compute the difference between the returns and the values
        delta = returns_tensor - values
        actor_loss = -(log_probs * delta.detach()).mean()
        critic_loss = delta.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss

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
