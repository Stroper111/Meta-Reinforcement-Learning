import numpy as np

from keras.utils import Sequence

from core.memory.sampling import AbstractSampling


class BaseSampling(AbstractSampling, Sequence):
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
        """ Fit model on a whole batch at once.  """
        batch_size = self.batch_size if batch_size is None else batch_size
        states, actions, rewards, terminal, states_next = self.replay_memory.get_batch(batch_size)

        q_values = self.model.predict(np.array(states))
        q_values_next = self.model.predict(np.array(states_next))

        targets = q_values.copy()
        for k in range(batch_size):
            q_update = rewards[k]
            if not terminal[k]:
                q_update = rewards[k] + self.gamma * np.amax(q_values_next[k])
            targets[k][actions[k]] = q_update

        model_input, model_output = states, targets
        return model_input, model_output

    def replay(self, batch_size):
        """ Fit model on every single data point.  """
        batch_size = self.batch_size if batch_size is None else batch_size
        batch = self.replay_memory.get_batch(batch_size)
        states, actions, rewards, terminal, states_next = batch

        q_values = self.model.predict(states)
        q_values_next = self.model.predict(states_next)

        targets = q_values.copy()
        for k in range(batch_size):
            q_update = rewards[k]
            if not terminal[k]:
                q_update = rewards[k] + self.gamma * np.amax(q_values_next[k])
            targets[k][actions[k]] = q_update

            state = np.expand_dims(states[k], 0)
            target = np.expand_dims(targets[k], 0)
            self.model.model.fit(state, target, verbose=0)
