import numpy as np

from copy import deepcopy
from keras.utils import Sequence

from .abstract import AbstractSampling
from core.memory import ReplayMemory


class BaseSampling(AbstractSampling, Sequence):
    """
        Base implementation of Sampling. Creates different batches of
        predefined batch sizes as a generator function or normal
        batches. All samples are taken randomly from the replay memory.

        This Sampling is meant to work together with the frame stacking
        hence there is a conversion before returning the values.

        batch_size: int
            The size of one batch
    """

    def __init__(self, replay_memory: ReplayMemory, batch_size: int = 64):
        super().__init__(replay_memory, batch_size)
        self.replay_memory = replay_memory
        self.batch_size = batch_size

    def __len__(self):
        return len(self.replay_memory) // self.batch_size

    def __getitem__(self, item):
        if item >= len(self):
            raise StopIteration

        sample_idx = np.random.randint(0, len(self.replay_memory), self.batch_size)
        model_input, model_output = self.replay_memory.get_batch(sample_idx)
        return self.reformat_states(model_input), model_output

    def random_batch(self, batch_size: int = None):
        """  Returns a single random batch.  """
        batch_size = self.batch_size if batch_size is None else batch_size
        sample_idx = np.random.randint(0, len(self.replay_memory), batch_size)
        model_input, model_output = self.replay_memory.get_batch(sample_idx)
        return self.reformat_states(model_input), model_output

    @staticmethod
    def reformat_states(states):
        """  Transforms the input of  stacked frame to the required format for the model.  """
        # Please always use deepcopy for this, since you use a lot of memory otherwise
        return np.array(deepcopy(states)).transpose([0, 2, 3, 1])
