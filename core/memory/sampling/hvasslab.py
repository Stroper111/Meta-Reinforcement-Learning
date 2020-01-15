import numpy as np

from copy import deepcopy
from keras.utils import Sequence

from .abstract import AbstractSampling


class SamplingHvassLab(AbstractSampling, Sequence):
    """
        Base implementation of Sampling. Creates different batches of
        predefined batch sizes as a generator function or normal
        batches. All samples are taken randomly from the replay memory.

        This Sampling is meant to work together with the frame stacking
        hence there is a conversion before returning the values.

        batch_size: int
            The size of one batch
    """

    def __init__(self, replay_memory, batch_size: int = 64):
        super().__init__(replay_memory, batch_size)
        self.replay_memory = replay_memory
        self.batch_size = batch_size

        self.error_threshold = 0.1

    def __len__(self):
        return len(self.replay_memory) // self.batch_size

    def __getitem__(self, item):
        if item >= len(self):
            raise StopIteration

        idx_low = np.random.choice(self.replay_memory.idx_error_low, size=self.replay_memory.num_samples_error_low, replace=False)
        idx_high = np.random.choice(self.replay_memory.idx_error_high, size=self.replay_memory.num_samples_error_high, replace=False)
        idx = np.concatenate((idx_low, idx_high))

        batch_states = np.transpose(np.array([self.replay_memory.states[loc] for loc in idx]), axes=(0, 3, 2, 1))
        batch_q_values = self.replay_memory.q_values[idx]
        return batch_states, batch_q_values

    def random_batch(self, batch_size: int = None):
        """  Returns a single random batch.  """
        old_batch_size = self.batch_size
        self.batch_size = batch_size
        data = self.__getitem__(0)
        self.batch_size = old_batch_size
        return data
