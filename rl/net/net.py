import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from .layers import NoisyLinear
import math
import numpy as np

USE_CUDA = torch.cuda.is_available()


class DQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape)

        return x


class DQN_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        #        self.fc2 = nn.Linear(hidden_size,hidden_size)
        #        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x, hx, cx):
        #        x = x.view(1,1,-1)
        #        hx = hx.view(1,1,-1)
        #        cx = cx.view(1,1,-1)

        x = F.leaky_relu(self.fc1(x))
        #        x = F.leaky_relu(self.fc2(x))
        #        hx, cx = self.lstm(x, (hx, cx))
        out, (hx, cx) = self.lstm(x, (hx, cx))
        #        hx = hx.view(hx.size(0),1,-1)
        #        cx = cx.view(cx.size(0),1,-1)

        x = F.leaky_relu(self.fc3(out))

        return x, hx, cx

    def parameter_update(self, source):
        self.load_state_dict(source.state_dict())


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super().__init__()
        self.state_space = state_size
        self.fc1 = nn.Linear(self.state_space, hidden_size)
        #        self.fc2 = nn.Linear(hidden_size,hidden_size)
        #        self.fc3 = nn.Linear(hidden_size,action_size)
        self.action_space = action_size
        self.fc_h = nn.Linear(hidden_size, hidden_size)
        self.fc_z_v = nn.Linear(hidden_size, 1)
        self.fc_z_a = nn.Linear(hidden_size, self.action_space)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        #        x = F.leaky_relu(self.fc2(x))
        #        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc_h(x))
        v, a = self.fc_z_v(x), self.fc_z_a(x)  # Calculate value and advantage streams
        a_mean = torch.stack(a.chunk(self.action_space, 1), 1).mean(1)
        x = v.repeat(1, self.action_space) + a - a_mean.repeat(1, self.action_space)
        return x


class NoisyDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(NoisyDQN, self).__init__()

        self.linear = nn.Linear(state_size, 128)
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, action_size)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        q_value = self.forward(state)
        return np.argmax(q_value.cpu().data.numpy())

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class CategoricalDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, **kwargs):
        super(CategoricalDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = kwargs.get('num_atoms', 51)
        self.v_min = kwargs.get('v_min', -10)
        self.v_max = kwargs.get('v_max', 10)

        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.noisy1 = NoisyLinear(128, 512)
        self.noisy2 = NoisyLinear(512, self.num_actions * self.num_atoms)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.v_min, self.v_max, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, **kwargs):
        super(RainbowDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = kwargs.get('num_atoms', 51)
        self.v_min = kwargs.get('v_min', -10)
        self.v_max = kwargs.get('v_max', 10)

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)

        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions,
                                            use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def act(self, state):
        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.v_min, self.v_max, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


class Reinforce(nn.Module):
    def __init__(self, num_inputs, action_space, hidden_size=256):
        super(Reinforce, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return F.softmax(action_scores)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist
