import numpy as np

from .abstract import AbstractSampling

from core.memory import ReplayMemory


class BaseSampling(AbstractSampling):
    """
        Base implementation of Sampling. Creates different batches of
        predefined batch sizes as a generator function or normal
        batches. All samples are taken randomly from the replay memory.

        batch_size: int
            The size of one batch
    """

    def __init__(self, replay_memory: ReplayMemory, batch_size: int = 64):
        super().__init__(replay_memory, batch_size)
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        pass

    def __len__(self):
        return len(self.replay_memory) // self.batch_size

    def __getitem__(self, item):
        if item >= len(self):
            raise StopIteration

        sample_idx = np.random.choice(range(len(self.replay_memory)), self.batch_size)
        model_input, model_output = self.replay_memory.get_batch(sample_idx)
        return model_input, model_output

    def random_batch(self, batch_size: None):
        """  Returns a single random batch.  """
        batch_size = self.batch_size if batch_size is None else batch_size
        sample_idx = np.random.choice(range(len(self.replay_memory)), batch_size)
        model_input, model_output = self.replay_memory.get_batch(sample_idx)
        return model_input, model_output
