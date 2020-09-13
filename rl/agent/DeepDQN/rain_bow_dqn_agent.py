from rl.agent.DeepDQN.dqn_agent import DQNAgent
from rl.net.net import RainbowDQN
import torch
from torch import optim
import numpy as np
from torch.autograd import Variable

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
USE_CUDA = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RainBowAgent(DQNAgent):
    def __init__(self, state_size, action_size, **kwargs):
        super(RainBowAgent, self).__init__(state_size, action_size, **kwargs)
        self.num_atoms = kwargs.get("num_atoms", 51)
        self.v_min = kwargs.get("v_max", -10)
        self.v_max = kwargs.get("v_max", 10)
        self.main_network = RainbowDQN(state_size, action_size, **kwargs).to(device)
        self.target_network = RainbowDQN(state_size, action_size, **kwargs).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state: (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection:

        Returns:

        """

        state = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            dist = self.main_network(state)
        dist = dist * torch.linspace(self.v_min, self.v_max, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action

    def train_step(self, experiences, gamma):
        """

        Args:
            experiences:
            gamma:

        Returns:

        """
        states = experiences['states']
        rewards = experiences['rewards']
        actions = experiences['actions']
        next_states = experiences['next_states']
        dones = experiences['dones']

        proj_dist = self.projection_distribution(next_states, rewards, dones)

        dist = self.main_network(states)
        action = actions.unsqueeze(1).unsqueeze(1).expand(BATCH_SIZE, 1, self.num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(Variable(proj_dist) * dist.log()).sum(1)
        loss = loss.mean()

        # Compute loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(self.main_network, self.target_network, TAU)
        self.main_network.reset_noise()
        self.target_network.reset_noise()

    def projection_distribution(self, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
        support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        next_dist = self.target_network(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1,
                                                                   next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        b = (Tz - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                      (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                      (next_dist * (b - l.float())).view(-1))

        return proj_dist
