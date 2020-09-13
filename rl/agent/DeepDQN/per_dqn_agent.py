from rl.agent.DeepDQN.dqn_agent import DQNAgent
from rl.memory.prioritized_replay import PrioritizedReplay
import torch
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
# LR = 5e-4  # learning rate
LR = 1e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
USE_CUDA = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PerDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super(PerDQNAgent, self).__init__(state_size, action_size)
        self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        td_error = self.get_td_error(reward, state, action, next_state)
        # self.memory.add(td, [state, action, reward, next_state, done])
        self.memory.add(state, action, reward, next_state, done, td_error.numpy())

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(type_data='tensor', shuffle=True)
            self.train_step(experiences, GAMMA)

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

        # Get expected Q values from local model
        q_values = self.main_network(states).gather(1, actions.view(-1, 1)).squeeze()

        # Get max predicted Q values (for next states) from target model
        next_q_values = self.target_network(next_states).detach().max(1)[0]
        # Compute Q targets for current states
        expected_q_value = rewards + (gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = F.mse_loss(q_values, expected_q_value)
        td_errors = expected_q_value - q_values
        # for i in range(BATCH_SIZE):
        #     val = abs(td_error[i].data[0])
        #     self.memory.update(batch_idx[i], val)
        self.memory.update_priorities(td_errors.detach().abs().numpy())

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
