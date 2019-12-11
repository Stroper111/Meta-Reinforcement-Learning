import numpy as np

from collections import deque, namedtuple

from core.preprocessing.wrappers import BaseWrapper


class Statistics(BaseWrapper):
    """ Converter for MultiEnv generated images.  """
    def __init__(self, env, save_dir):
        super(Statistics, self).__init__(env)
        self.save_dir = save_dir
        self.setup = env.setup
        self.instances = env.instances
        self.continuous_history_size = 30

        self._episodic = Episode(self.instances)
        self._continuous = Continuous(self.instances, self.continuous_history_size)

    def step(self, action):
        images, rewards, dones, infos = self.env.step(action)
        self._step_update(rewards, dones)
        return images, rewards, dones, infos

    def summary(self, stats: list = None):
        """ Print the information of the last statistics, stats have to be valid numpy functions.  """
        episodic = self._episodic.summary()
        continuous = self._continuous.summary(stats)
        return episodic, continuous

    def _step_update(self, rewards, dones):
        """ Update all statics on a step.  """
        self._episodic.update(rewards, dones)
        self._continuous.update(rewards, dones)


class Episode:
    def __init__(self, instances):
        self.episode = np.zeros(instances)
        self.steps = np.zeros(instances)
        self.rewards = np.zeros(instances)

    def update(self, rewards, dones):
        self.rewards += rewards
        self.episode[dones] += 1
        self.steps[np.bitwise_not(dones)] += 1

        self.steps[dones] = 0
        self.rewards[dones] = 0

    def summary(self):
        return dict(episode=self.episode, steps=self.steps, rewards=self.rewards)


class Continuous:
    def __init__(self, instances, history_size):
        self._data = [deque([0], maxlen=history_size) for _ in range(instances)]

    def update(self, rewards, dones):
        [game.append(data) for game, data in zip(self._data, rewards)]

    def summary(self, stats):
        data = dict()
        for stat in stats:
            data[stat] = [getattr(np, stat)(game) for game in self._data]
        return data


class Log:
    def __init__(self, save_dir):
        pass

    def write(self):
        pass

    def read(self):
        pass
