import numpy as np

from copy import deepcopy
from keras.utils import Sequence

from .abstract import AbstractSampling


class BaseSamplingMultiEnv(AbstractSampling, Sequence):
    """
        Base implementation of Sampling. Creates different batches of
        predefined batch sizes as a generator function or normal
        batches. All samples are taken randomly from the replay memory.

        This Sampling is meant to work together with the frame stacking
        hence there is a conversion before returning the values.

        batch_size: int
            The size of one batch
    """

    def __init__(self, replay_memory, model, alpha, gamma, batch_size: int = 64):
        super().__init__(replay_memory, batch_size)
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

    def __len__(self):
        return len(self.replay_memory) // self.batch_size

    def __getitem__(self, item):
        if item >= len(self):
            raise StopIteration
        return self.random_batch(self.batch_size)

    def random_batch(self, batch_size: int = None):
        """  Returns a single random batch.  """
        batch_size = self.batch_size if batch_size is None else batch_size
        sample_idx = np.random.randint(0, len(self.replay_memory)-1, batch_size)

        states, actions, rewards, done, states_next = self.replay_memory.get_batch(sample_idx)
        model_input, model_output = self._create_model_input(states, actions, rewards, done, states_next)
        return model_input, model_output

    def _create_model_input(self, states, actions, rewards, done, states_next):
        states = self.reformat_states(states)
        states_next= self.reformat_states(states_next)

        q_values = self.model.predict(states)
        q_values_next = self.model.predict(states_next)

        targets = []
        for k in range(self.batch_size):
            target = rewards[k]
            if not done[k]:
                target = rewards[k] + self.alpha * (self.gamma * np.amax(q_values_next[k]))

            target_f = q_values[k]
            target_f[actions[k]] = target
            targets.append(target_f)

        model_input, model_output = states, np.array(targets)
        return model_input, model_output

    @staticmethod
    def reformat_states(states):
        """  Transforms the input of  stacked frame to the required format for the model.  """
        # Please always use deepcopy for this, since you use a lot of memory otherwise
        return np.array(deepcopy(states)).transpose([0, 2, 3, 1])
