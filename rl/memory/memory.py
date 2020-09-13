import numpy as np
from collections import deque
import operator
import torch
from texttable import Texttable
from abc import abstractmethod


class Memory(object):
    def __init__(self, buffer_size, batch_size, data_keys=None, **kwargs):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.is_episodic = False
        self.batch_indices = None
        self.head = -1
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        if data_keys is not None:
            self.data_keys = data_keys
        else:
            self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.reset()

    def reset(self):
        """
        Initializes the memory arrays, size and head pointer

        Returns:

        """
        self.head = -1
        self.size = 0
        for k in self.data_keys:
            setattr(self, k, [None] * self.buffer_size)

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        """

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:

        """

    @abstractmethod
    def add_experience(self, state, action, reward, next_state, done):
        """
        Implementation for update() to add experience to memory,
        expanding the memory size if necessary

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, shuffle=False, clear=False, type_data="np", **kwargs):
        """

        Returns: a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the
        corresponding sampled elements
        """
        raise NotImplementedError

    def convert_data_type(self, batch, data_type):
        """

        Args:
            batch:
            data_type:

        Returns:

        """
        if data_type == 'np':
            for k in self.data_keys:
                batch[k] = np.array(batch[k])
        elif data_type == 'tensor':
            for k in self.data_keys:
                if k in ["states", "next_states", 'rewards']:
                    batch[k] = torch.tensor(batch[k]).float()
                elif k in ['actions', 'dones']:
                    batch[k] = torch.tensor(batch[k]).long()

    def __len__(self):
        return self.size

    def __str__(self):
        tee = Texttable(max_width=150)
        tee.add_rows([["MEMORY DATA INFORMATION", "Size", "sample data[:5]"]])
        for k in self.data_keys:
            info = np.array(getattr(self, k))
            tee.add_row([k, info.shape, info[:5]])
        return tee.draw()
