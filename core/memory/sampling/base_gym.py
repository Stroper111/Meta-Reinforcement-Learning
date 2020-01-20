import numpy as np

from keras.utils import Sequence

from .abstract import AbstractSampling


class BaseSamplingGym(AbstractSampling, Sequence):
    """
        Base implementation of Sampling. Creates different batches of
        predefined batch sizes as a generator function or normal
        batches. All samples are taken randomly from the replay memory.

        This Sampling is meant to work together with the frame stacking
        hence there is a conversion before returning the values.

        replay_memory
            This is the memory from which the samples are extracted it should
            be compatible with Keras.Sequence.

        model: Keras
            This should be a Keras model that is used to make predictions with.

        gamma: float
            Discount factor for the reinforcement learning update rule.

        alpha: float
            Learning rate for the target

        batch_size: int
            The size of one batch
    """

    def __init__(self, replay_memory, model, gamma, alpha=0.1, batch_size: int = 64):
        super().__init__(replay_memory, batch_size)
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.model = model
        self.gamma = gamma
        self.alpha = alpha

    def __len__(self):
        return len(self.replay_memory) // self.batch_size

    def __getitem__(self, item):
        if item >= len(self):
            raise StopIteration

        return self.random_batch()

    def random_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        sample_idx = np.random.randint(0, len(self.replay_memory) - 1, self.batch_size)
        states, actions, rewards, done, states_next = self.replay_memory.get_batch(sample_idx)

        q_values = self.model.predict(states)
        q_values_next = self.model.predict(states_next)

        targets = []
        for k in range(batch_size):
            target = rewards[k]
            if not done[k]:
                target = rewards[k] + self.alpha * (self.gamma * np.amax(q_values_next[k]))

            target_f = q_values[k]
            target_f[actions[k]] = target
            targets.append(target_f)

        model_input, model_output = states, np.array(targets)
        return model_input, model_output
