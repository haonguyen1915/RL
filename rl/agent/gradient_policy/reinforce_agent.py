from rl.net.net import Reinforce, ActorCritic
from rl.memory.memory import Buffer
import numpy as np
import torch
import os
from torch import optim
from rl.agent.agent import Agent

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 1  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
USE_CUDA = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReinforceAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, **kwargs):
        """
        Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        super(ReinforceAgent, self).__init__(state_size, action_size, **kwargs)
        # Replay memory
        self.memory = Buffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        self.network = Reinforce(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.b_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.train_step(experiences, GAMMA)

    def batch_step(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(
                batch_state, batch_action, batch_reward, batch_next_state, batch_done
        ):
            self.memory.add(state, action, reward, next_state, done)
        self.b_step += 1
        # Learn every UPDATE_EVERY time steps.
        if self.b_step % BATCH_SIZE == 0:
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

        action_probs = self.network(torch.FloatTensor(state)).detach().numpy()
        action = np.random.choice(self.action_size, p=action_probs)
        return action

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma: discount factor

        Returns:

        """
        states, actions, rewards, next_states, dones = experiences
        rewards = rewards.numpy().squeeze()

        # Compute Gt or returns
        returns = self.compute_returns(rewards, 1.0, gamma, dones)
        mean = np.mean(returns)
        std = np.std(returns) if np.std(returns) > 0 else 1
        returns = (returns - mean) / std
        returns_tensor = torch.tensor(returns)

        # Calculate loss
        loss = torch.tensor([0.0])
        log_probs = torch.log(self.network(states)).squeeze()
        log_probs = log_probs.gather(1, actions)
        for g, log_prob in zip(returns_tensor, log_probs):
            loss += -g * log_prob

        # Calculate gradients
        loss.backward()
        # Apply gradients
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


class A2CAgent(ReinforceAgent):
    def __init__(self, state_size, action_size, **kwargs):
        super(A2CAgent, self).__init__(state_size, action_size, **kwargs)
        self.network = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:

        """

        state = torch.tensor(state).unsqueeze(0).float()
        value, policy_dist = self.network(state)
        dist = policy_dist.detach().numpy()

        value = value.detach().numpy()[0, 0]
        action = np.random.choice(self.action_size, p=np.squeeze(dist))
        return value, action

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma: discount factor

        Returns:

        """
        states, actions, rewards, next_states, dones = experiences
        rewards = rewards.numpy().squeeze()

        values, dist_actions = self.network(states)
        values = values.squeeze(1).detach().numpy()
        log_probs = torch.log(dist_actions.gather(1, actions)).squeeze()

        next_state = next_states[-1]
        next_value, _ = self.network(next_state.unsqueeze(0))
        next_value = next_value.detach().numpy()[0, 0]

        returns = self.compute_returns(next_value, rewards, gamma)

        # update actor critic
        values = torch.tensor(values)
        returns = torch.tensor(returns)

        advantage = returns - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * 0
        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()
