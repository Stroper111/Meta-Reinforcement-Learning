
from abc import ABC

from core.memory import ReplayMemory

class AbstractSampling(ABC):
    """
        Abstract class to unify all different sampling methods.
        This class is a generator to work with keras generator.

        replay_memory: ReplayMemory
            This is used to extract samples from.
    """
    def __init__(self, replay_memory: ReplayMemory):
        pass

    def __next__(self):
        pass

    def batch(self):
        pass
