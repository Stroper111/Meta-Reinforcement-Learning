import random
import numpy as np

from collections import namedtuple, deque


class BaseMemoryDeque:
    """
    A Base implementation of a deque memory.

    :param size: int
        The maximum size of the memory
    :param to_numpy: bool
        Flag variable to indicate a conversion to numpy is preferred whenever returning a batch.
        This is required whenever the input data is a vector. (e.g. reward = [1])
    """
    _transition = namedtuple('transition', ('state', 'action', 'reward', 'done', 'next_state'))

    def __init__(self, size, to_numpy=True):
        self.size = size
        self.to_numpy = to_numpy

        self.pointer = 0
        self.memory = deque(maxlen=size)

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.size

    def reset(self):
        self.pointer = 0
        self.filled = False

    def get_batch(self, batch_size=32):
        states, actions, rewards, done, states_next = zip(*random.sample(self.memory, batch_size))

        if self.to_numpy:
            states = np.vstack(states)
            actions = np.hstack(actions)
            rewards = np.hstack(rewards)
            done = np.hstack(done)
            states_next = np.vstack(states_next)

        return self._transition(states, actions, rewards, done, states_next)

    def add(self, state, action, reward, done, next_state):
        self.memory.append(self._transition(state, action, reward, done, next_state))
