import numpy as np

from keras.utils import Sequence

from core.memory.sampling import AbstractSampling


class PolicyGradientsSampling(AbstractSampling, Sequence):
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
        states, actions, rewards, done, states_next = self.replay_memory.get_batch(batch_size)

        # This is required for the tuple based memory.
        if not isinstance(states, np.ndarray):
            states = np.vstack(states)

        # TODO Initialize with zeros instead of redundant predictions
        preds = self.model.predict(states)
        targets = preds.copy()

        running_add = rewards[-1]
        disc_reward = 0
        for t, k in enumerate(range(batch_size)):
            if not done[k]:
                disc_reward = running_add * (self.gamma ** rewards[t])
                # Normalize
                disc_reward -= np.mean(disc_reward)
                disc_reward /= np.std(disc_reward)

            targets[k][actions[k]] = disc_reward
        model_input, model_output = states, targets
        return model_input, model_output

    def replay(self, batch_size):
        batch_size = self.batch_size if batch_size is None else batch_size
        batch = self.replay_memory.get_batch(batch_size)
        states, actions, rewards, terminal, states_next = batch

        if actions[0].ndim == 1:
            states = np.vstack(states)
            rewards = np.hstack(rewards)
            terminal = np.hstack(terminal)

        targets = self.process_rewards(rewards)
        for k in range(batch_size):
            if not terminal:
                state = np.expand_dims(states[k], 0)
                target = np.expand_dims(targets[k], 0)
                self.model.model.fit(state, target, verbose=0)

    def process_rewards(self, reward_list):
        """Returns Discounted rewards"""
        rewards = np.vstack(reward_list)
        discounted_rewards = np.zeros_like(rewards)
        running_add = rewards[-1]
        for t in range(len(rewards)):
            discounted_rewards[t] = running_add * (self.gamma ** t)

        # Normalize
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        return discounted_rewards
