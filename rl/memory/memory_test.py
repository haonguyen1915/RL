from rl.net.net import DQN
import numpy as np
import random
import time
from copy import deepcopy
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed=777):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state",
                                                  "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory

        Returns:
            states: (batch_size, num_state]
            actions: (batch_size, action]
            rewards: (batch_size, reward]
            next_states: (batch_size, num_state]
            dones: (batch_size, done]

        """
        experiences = deepcopy(self.memory)
        if experiences[0].state.shape[0] == 1:
            states = torch.from_numpy(
                np.vstack([np.expand_dims(e.state, axis=0) for e in experiences if
                           e is not None])).float().to(device)
            next_states = torch.from_numpy(
                np.vstack([np.expand_dims(e.next_state, axis=0) for e in experiences if
                           e is not None])).float().to(device)
        else:
            states = torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        self.clear()
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.memory = deque(maxlen=self.buffer_size)

    @staticmethod
    def discount_rewards(rewards, gamma=0.99):
        r = np.array([gamma ** i * rewards[i]
                      for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        return r - r.mean()

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed=777):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state",
                                                  "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory

        Returns:
            states: (batch_size, num_state]
            actions: (batch_size, action]
            rewards: (batch_size, reward]
            next_states: (batch_size, num_state]
            dones: (batch_size, done]

        """
        experiences = random.sample(self.memory, k=self.batch_size)
        if experiences[0].state.shape[0] == 1:
            states = torch.from_numpy(
                np.vstack([np.expand_dims(e.state, axis=0) for e in experiences if
                           e is not None])).float().to(device)
            next_states = torch.from_numpy(
                np.vstack([np.expand_dims(e.next_state, axis=0) for e in experiences if
                           e is not None])).float().to(device)
        else:
            states = torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.memory = deque(maxlen=self.buffer_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PER_Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def push(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch_idx = []
        batch = []

        segment = self.tree.total() / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            print(f"s: {s}")
            print(f"self.tree.tree: {self.tree.tree[0]}")
            batch.append(data)
            batch_idx.append(idx)
        [states, actions, rewards, next_states, dones] = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.LongTensor(dones).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        batch = (states, actions, rewards, next_states, dones)

        return batch, batch_idx

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return int(self.tree.total())
