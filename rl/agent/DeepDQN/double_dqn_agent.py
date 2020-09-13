from rl.agent.DeepDQN.dqn_agent import DQNAgent
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


class DoubleDQNAgent(DQNAgent):
    def __init__(self, state_size, action_size):
        super(DoubleDQNAgent, self).__init__(state_size, action_size)

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
        # states, actions, rewards, next_states, dones = experiences
        # Get expected Q values from local model
        q_values = self.main_network(states).gather(1, actions.view(-1, 1)).squeeze()

        # Get max predicted Q values (for next states) from target model
        argmax_Q = self.main_network(next_states).max(1)[1].view(-1, 1)

        # Compute Q targets for current states
        # next_q_values = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, argmax_Q).squeeze()
        expected_q_value = rewards + (gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = F.mse_loss(q_values, expected_q_value)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.main_network, self.target_network, TAU)
