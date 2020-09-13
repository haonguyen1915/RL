import random
import numpy as np
from rl.memory.replay import Replay


class SumTree:
    """
    Helper class for PrioritizedReplay

    This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

    MIT License

    Copyright (c) 2018 Jaromír Janisch

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences

    def _propagate(self, idx, change):
        """
        For each node at index i, the
            - left child is at index 2*i+1,
            - right child at 2*i+2
            - and the parent is at (i-1)//2

        Args:
            idx:
            change:

        Returns:

        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        For each node at index i, the left child is at index 2*i+1,
        right child at 2*i+2 and the parent is at (i-1)//2

        Args:
            idx:
            s:

        Returns:

        """
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

    def add(self, p, index):
        """

        Args:
            p: priority
            index: data index

        Returns:

        """
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        """

        Args:
            idx: tree index
            p: priority

        Returns:

        """
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.indices[indexIdx]

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')


class PrioritizedReplay(Replay):

    def __init__(self, buffer_size, batch_size, data_keys=None, **kwargs):
        """
        Prioritized Experience Replay

        Implementation follows the approach in the paper "Prioritized Experience Replay",
        Schaul et al 2015" https://arxiv.org/pdf/1511.05952.pdf and is Jaromír Janisch's with minor
        adaptations. See memory_util.py for the license and link to Jaromír's excellent blog

        Stores agent experiences and samples from them for agent training according to each
        experience's priority

        The memory has the same behaviour and storage structure as Replay memory with the addition
        of a SumTree to store and sample the priorities.
        Args:
            memory_spec:
            body:
        """
        super().__init__(buffer_size, batch_size, data_keys)
        self.tree_indices = None
        self.batch_indices = None
        self.tree = SumTree(self.buffer_size)
        self.epsilon = np.full((1,), kwargs.get("epsilon", 0.01))
        self.alpha = np.full((1,), kwargs.get("alpha", 0.6))
        # adds a 'priorities' scalar to the data_keys and call reset again
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']
        self.reset()

    def reset(self):
        super().reset()
        self.tree = SumTree(self.buffer_size)

    def add(self, state, action, reward, next_state, done, error=100000):
        """
        Implementation for update() to add experience to memory, expanding the memory size
        if necessary.
        All experiences are added with a high priority to increase the likelihood that they are
        sampled at least once.

        Args:
            state:
            action:
            reward:
            next_state:
            done:
            error:

        Returns:

        """
        super().add_experience(state, action, reward, next_state, done)

        priority = self.get_priority(error)
        getattr(self, 'priorities')[self.head] = priority
        self.tree.add(priority, self.head)

    def get_priority(self, error):
        """
        Takes in the error of one or more examples and returns the proportional priority

        Args:
            error:

        Returns:

        """
        return np.power(error + self.epsilon, self.alpha).squeeze()

    def sample_indices(self, batch_size, shuffle=None):
        """
        Samples batch_size indices from memory in proportional to their priority.

        Args:
            batch_size:
            shuffle:

        Returns:

        """
        batch_indices = np.zeros(batch_size)
        tree_indices = np.zeros(batch_size, dtype=np.int)

        for i in range(batch_size):
            s = random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_indices[i] = idx
            tree_indices[i] = tree_idx

        batch_indices = np.asarray(batch_indices).astype(int)
        self.tree_indices = tree_indices
        return batch_indices

    def update_priorities(self, errors):
        """
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_indices

        Args:
            errors:

        Returns:

        """
        priorities = self.get_priority(errors)
        assert len(priorities) == self.batch_indices.size

        for index, p in zip(self.batch_indices, priorities):
            getattr(self, 'priorities')[index] = p
        for p, i in zip(priorities, self.tree_indices):
            self.tree.update(i, p)


if __name__ == "__main__":
    tree = SumTree(10)
    for idx in range(10):
        tree.add(idx*0.1, idx)

    out = []
    for _ in range(10):
        s = random.uniform(0, tree.total())
        r = tree.get(s)
        out.append(r[2])

    print(f"tree: {tree.tree}")
    print(f"indices: {tree.indices}")
    print(f"total: {tree.total()}")
    print(f"out: {out}")
    tree.print_tree()
