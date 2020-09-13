import numpy as np
from collections import deque
import torch


class OnPolicyReplay(object):
    '''
    Stores agent experiences and returns them in a batch for agent training.

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory does not have a fixed size. Instead the memory stores data from N episodes,
    where N is determined by the user. After N episodes, all of the examples are
    returned to the agent to learn from.

    When the examples are returned to the agent, the memory is cleared to prevent the agent
    from learning from off policy experiences. This memory is intended for on policy algorithms.

    Differences vs. Replay memory:
        - Experiences are nested into episodes. In Replay experiences are flat, and episode is not tracked
        - The entire memory constitues a batch. In Replay batches are sampled from memory.
        - The memory is cleared automatically when a batch is given to the agent.

    e.g. memory_spec
    "memory": {
        "name": "OnPolicyReplay"
    }
    '''

    def __init__(self, buffer_size, data_keys=None):
        # super().__init__()
        # NOTE for OnPolicy replay, frequency = episode; for other classes below frequency = frames
        # util.set_attr(self, self.body.agent.agent_spec['algorithm'], ['training_frequency'])
        self.training_frequency = 1
        self.buffer_size = buffer_size
        # Don't want total experiences reset when memory is
        self.is_episodic = True
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        # declare what data keys to store
        if data_keys is not None:
            self.data_keys = data_keys
        else:
            self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.cur_epi_data = {}
        self.most_recent = {}
        self.reset()

    def reset(self):
        """
        Resets the memory. Also used to initialize memory vars

        Returns:

        """
        for k in self.data_keys:
            setattr(self, k, deque(maxlen=self.buffer_size))
        self.cur_epi_data = {k: [] for k in self.data_keys}
        self.most_recent = (None,) * len(self.data_keys)
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """
        Interface method to update memory

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
        Interface helper method for update() to add experience to memory

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:

        """
        self.most_recent = (state, action, reward, next_state, done)
        for idx, k in enumerate(self.data_keys):
            self.cur_epi_data[k].append(self.most_recent[idx])
        # If episode ended, add to memory and clear cur_epi_data
        if np.isscalar(done) and done:
            for k in self.data_keys:
                getattr(self, k).append(self.cur_epi_data[k])
            self.cur_epi_data = {k: [] for k in self.data_keys}
            # If agent has collected the desired number of episodes, it is ready to train
            # length is num of epis due to nested structure
            # if len(self.states) == self.body.agent.algorithm.training_frequency:
        # Track memory size and num experiences
        self.size += 1
        self.seen_size += 1

    def get_most_recent_experience(self):
        """
        Returns the most recent experience

        Returns:

        """
        return self.most_recent

    def sample(self, type_data="np"):
        """
            Returns all the examples from memory in a single batch. Batch is stored as a dict.
            Keys are the names of the different elements of an experience. Values are nested lists of
            the corresponding sampled elements. Elements are nested into episodes
            e.g.
            batch = {
                'states'     : [[s_epi1], [s_epi2], ...],
                'actions'    : [[a_epi1], [a_epi2], ...],
                'rewards'    : [[r_epi1], [r_epi2], ...],
                'next_states': [[ns_epi1], [ns_epi2], ...],
                'dones'      : [[d_epi1], [d_epi2], ...]}

            Returns: batch

        """
        if type_data == "np":
            batch = {k: np.array(getattr(self, k)) for k in self.data_keys}
        elif type_data == "tensor":
            batch = {k: torch.tensor(getattr(self, k)) for k in self.data_keys}
        else:
            batch = {k: getattr(self, k) for k in self.data_keys}
        self.reset()
        return batch

    def __str__(self):
        return str(self.cur_epi_data)

    def __len__(self):
        return len(getattr(self, self.data_keys[0]))
