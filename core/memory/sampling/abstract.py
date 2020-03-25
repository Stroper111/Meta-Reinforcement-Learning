import numpy as np

from abc import ABC, abstractmethod
from typing import Union


class AbstractSampling(ABC):
    """
        Abstract class to unify all different sampling methods.
        This class is a generator to work with keras generator.

        replay_memory: ReplayMemory
            This is the memory from which samples are extracted.
        batch-size: int
            The number of samples per batch.
    """

    @abstractmethod
    def __init__(self, replay_memory, batch_size: int):
        pass

    @abstractmethod
    def __len__(self):
        """  This will determine how many times the keras generator will loop over the dataset.  """
        pass

    @abstractmethod
    def __getitem__(self, item) -> (np.array, np.array):
        """ Return generator X and Y.  """
        pass

    @abstractmethod
    def random_batch(self, batch_size: Union[None, int]) -> (np.array, np.array):
        """ Return one single batch X and Y.  """
        pass
