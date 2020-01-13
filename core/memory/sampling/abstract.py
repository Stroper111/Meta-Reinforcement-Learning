import numpy as np

from abc import ABC

from core.memory import ReplayMemory


class AbstractSampling(ABC):
    """
        Abstract class to unify all different sampling methods.
        This class is a generator to work with keras generator.

        replay_memory: ReplayMemory
            This is used to extract samples from.
        batch-size: int
            The number of samples per batch.
    """

    def __init__(self, replay_memory: ReplayMemory, batch_size: int = 64):
        pass

    def __len__(self):
        """  This will determine how many times the keras generator will loop over the dataset.  """
        pass

    def __getitem__(self, item) -> (np.array, np.array):
        """ Return generator X and Y.  """
        pass
