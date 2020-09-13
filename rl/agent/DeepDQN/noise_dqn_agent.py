from rl.agent.DeepDQN.dqn_agent import DQNAgent
from rl.net.net import NoisyDQN
import torch
from torch import optim
import numpy as np

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
USE_CUDA = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NoiseDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
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
