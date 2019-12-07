
import numpy as np

from collections import deque


class ReplayMemory:
    def __init__(self, size):
        self.size = size
        self.memory = np.zeros(size)

        self.in_use = 0

    def __getitem__(self, item):
        return self.memory[item]

    def __len__(self):
        return self.in_use

