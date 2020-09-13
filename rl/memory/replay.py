import numpy as np
from collections import deque
import operator
from rl.memory.memory import Memory


class Replay(Memory):
    """
    Stores agent experiences and samples from them for agent training

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting experiences are O(1)
        - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    If 'use_cer', sampling will add the latest experience.

    e.g. memory_spec
    "memory": {
        "name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    """

    def __init__(self, buffer_size, batch_size, data_keys=None):
        super(Replay, self).__init__(buffer_size, batch_size, data_keys)
        self.user_cer = False

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
        self.add_experience(state, action, reward, next_state, done)

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
        self.head = (self.head + 1) % self.buffer_size
        most_recent = (state, action, reward, next_state, done)
        for idx, k in enumerate(self.data_keys):
            # getattr(self, k).append(most_recent[idx])
            if idx < len(most_recent):
                getattr(self, k)[self.head] = most_recent[idx]

        if self.size < self.buffer_size:
            self.size += 1
        self.seen_size += 1

    def sample(self, shuffle=False, clear=False, type_data=None, **kwargs):
        """

        Returns: a batch of batch_size samples. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the
        corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones}

        Args:
            clear: Clear data once sampled
            shuffle: Shuffle all data
            type_data: should be [np, tensor]

        """
        batch = {}
        if shuffle:
            self.batch_indices = self.sample_indices(self.batch_size)
            for k in self.data_keys:
                batch[k] = self.batch_get(getattr(self, k), self.batch_indices)
        else:
            for k in self.data_keys:
                all_data = getattr(self, k)[:self.size]
                bs = min(self.size, self.batch_size)
                batch[k] = all_data[-bs:]
        self.convert_data_type(batch, type_data)

        if clear:
            self.reset()

        return batch

    def sample_indices(self, batch_size):
        """
        Batch indices a sampled random uniformly

        Args:
            batch_size:

        Returns:

        """
        batch_indices = np.random.randint(len(self), size=batch_size)

        if self.user_cer:
            batch_indices[-1] = self.head

        return batch_indices

    @staticmethod
    def batch_get(arr, indices):
        """
        Get multi-idxs from an array depending if it's a python list or np.array

        Args:
            arr:
            indices:

        Returns:

        """
        if isinstance(arr, (list, deque)):
            return np.array(operator.itemgetter(*indices)(arr))
        else:
            return arr[indices]

