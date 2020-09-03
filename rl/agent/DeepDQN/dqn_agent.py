from rl.net.net import DQN, DuelingDQN, NoisyDQN, RainbowDQN, CategoricalDQN
from rl.memory.memory import ReplayBuffer, PER_Memory
import numpy as np
import random
import torch
import os
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from rl.agent.agent import Agent

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
USE_CUDA = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, **kwargs):
        """
        Initialize an Agent object.

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        super(DQNAgent, self).__init__(state_size, action_size, **kwargs)
        self.state_size = state_size
        self.action_size = action_size

        # Q-Network
        self.main_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

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

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:

        """
        if isinstance(state, (np.int64, int)):
            state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.main_network(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma: discount factor

        Returns:

        """
        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        next_q_value = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        expected_q_value = rewards + (gamma * next_q_value * (1 - dones))

        # Get expected Q values from local model
        q_value = self.main_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(self.main_network, self.target_network, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau: interpolation parameter

        Returns:

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, model_path):
        if not os.path.dirname(model_path):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.main_network.state_dict(), model_path)

    def load(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"File not found: {model_path}")
        self.main_network.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')))


class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, seed):
        super(DoubleDQNAgent, self).__init__(state_size, action_size)

    def train_step(self, experiences, gamma):
        """

        Args:
            experiences:
            gamma:

        Returns:

        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        argmax_Q = self.main_network(next_states).max(1)[1].unsqueeze(1)
        # Compute Q targets for current states
        # next_q_values = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, argmax_Q)
        expected_q_value = rewards + (gamma * next_q_values * (1 - dones))

        # Get expected Q values from local model
        q_values = self.main_network(states).gather(1, actions)
        # print(q_values.size())
        # print(actions.size())

        # Compute loss
        loss = F.mse_loss(q_values, expected_q_value)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.main_network, self.target_network, TAU)


class PerDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, seed):
        super(PerDQNAgent, self).__init__(state_size, action_size, seed)
        self.memory = PER_Memory(BUFFER_SIZE)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        td = self.get_td_error(reward, state, action, next_state)
        self.memory.push(td, [state, action, reward, next_state, done])
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.train_step(experiences, GAMMA)

    def train_step(self, experiences, gamma):
        """

        Args:
            experiences:
            gamma:

        Returns:

        """
        batch, batch_idx = experiences
        states, actions, rewards, next_states, dones = batch

        # Get max predicted Q values (for next states) from target model
        next_q_values = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        expected_q_value = rewards + (gamma * next_q_values * (1 - dones))

        # Get expected Q values from local model
        q_values = self.main_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_values, expected_q_value)
        td_error = expected_q_value - q_values
        # print(td_error)
        # exit()
        for i in range(BATCH_SIZE):
            val = abs(td_error[i].data[0])
            self.memory.update(batch_idx[i], val)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.main_network, self.target_network, TAU)

    def get_td_error(self, re, st, ac, st_):
        re = torch.FloatTensor([re]).view(1, -1)
        st = torch.FloatTensor([st]).view(1, -1)
        st_ = torch.FloatTensor([st_]).view(1, -1)
        ac = torch.LongTensor([ac]).view(1, -1)

        td_error = re + GAMMA * self.target_network(st_).max(1)[0] - self.main_network(
            st).gather(1, ac)
        return abs(td_error.data[0][0])


class DualDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, seed):
        super(DualDQNAgent, self).__init__(state_size, action_size)
        self.main_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)


class DualDoubleDQNAgent(DoubleDQNAgent):
    def __init__(self, state_size, action_size, seed):
        super(DualDoubleDQNAgent, self).__init__(state_size, action_size, seed)
        self.main_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)


class NoiseDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, seed):
        super(NoiseDQNAgent, self).__init__(state_size, action_size)
        self.main_network = NoisyDQN(state_size, action_size).to(device)
        self.target_network = NoisyDQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.count_step = 0

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        Args:
            state: (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection:

        Returns:

        """

        state = torch.tensor(state).float()
        with torch.no_grad():
            q_value = self.main_network(state)
        return np.argmax(q_value.cpu().data.numpy())

    def train_step(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma: discount factor

        Returns:

        """
        super().train_step(experiences, gamma)

        self.main_network.reset_noise()
        self.target_network.reset_noise()


class DistPerspectiveDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size, **kwargs):
        super(DistPerspectiveDQNAgent, self).__init__(state_size, action_size, **kwargs)
        self.num_atoms = kwargs.get("num_atoms", 51)
        self.v_min = kwargs.get("v_max", -10)
        self.v_max = kwargs.get("v_max", 10)
        self.main_network = CategoricalDQN(state_size, action_size, **kwargs).to(device)
        self.target_network = CategoricalDQN(state_size, action_size, **kwargs).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)

    def act(self, state, eps=None):
        # state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        # dist = self.forward(state).data.cpu()
        # dist = dist * torch.linspace(self.v_min, self.v_max, num_atoms)
        # action = dist.sum(2).max(1)[1].numpy()[0]
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
        states, actions, rewards, next_states, dones = experiences

        actions = actions.squeeze(1)
        dones = dones.squeeze(1)
        rewards = rewards.squeeze(1)
        proj_dist = self.projection_distribution(next_states, rewards, dones)

        dist = self.main_network(states)
        action = actions.unsqueeze(1).unsqueeze(1).expand(BATCH_SIZE, 1, self.num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = - (Variable(proj_dist) * dist.log()).sum(1).mean()

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
        states, actions, rewards, next_states, dones = experiences

        actions = actions.squeeze(1)
        dones = dones.squeeze(1)
        rewards = rewards.squeeze(1)
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
