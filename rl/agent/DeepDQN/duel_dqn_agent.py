from rl.agent.DeepDQN.dqn_agent import DQNAgent
from rl.net.net import DuelingDQN
import torch
from torch import optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
USE_CUDA = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DualDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super(DualDQNAgent, self).__init__(state_size, action_size)
        self.main_network = DuelingDQN(state_size, action_size).to(device)
        self.target_network = DuelingDQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=LR)
