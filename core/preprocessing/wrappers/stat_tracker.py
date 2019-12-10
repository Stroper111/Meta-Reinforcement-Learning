import numpy as np

from collections import deque, namedtuple

from core.preprocessing.wrappers import BaseWrapper


class Statistics(BaseWrapper):
    """ Converter for MultiEnv generated images.  """
    aggregate_ = namedtuple('aggregate', ('n', 'min', 'q1', 'median', 'q3', 'max', 'mean'))
    episodic_ = namedtuple('episodic', ('episode', 'steps', 'rewards'))
    def __init__(self, env, save_dir):
        super(Statistics, self).__init__(env)
        self.save_dir = save_dir
        self.setup = env.setup
        self.instances = env.instances
        self.continuous_backlog = 30

        self._episodic = Episode(self.instances)
        self._continuous = self._create_data_continuous()

    def step(self, action):
        images, rewards, dones, infos = self.env.step(action)
        self._step_update(rewards, dones)
        return images, rewards, dones, infos

    def print_statistics(self, stats: list = None):
        """ Print the information of the last statistics, stats have to be valid numpy functions.  """

        # Get the stats for every instance
        per_instance = {stat: [getattr(np, stat)(game) for game in self._continuous] for stat in stats}
        return per_instance, self._episodic

    def _create_data_continuous(self):
        """ Create all continuous data, the dict key has to be valid numpy values.  """
        data = [deque([0], maxlen=self.continuous_backlog) for _ in range(self.instances)]
        return data

    def _data_game(self):
        """ Combines the data of all instances of a single game.  """
        pass

    def _step_update(self, rewards, dones):
        """ Update all statics on a step.  """
        self._step_update_continuous(rewards, dones)
        self._step_update_episodic(rewards, dones)
        pass

    def _step_update_continuous(self, rewards, dones):
        """ Update all statics on a step.  """
        for game, data in zip(self._continuous, rewards):
            game.append(data)

    def _step_update_episodic(self, rewards, dones):
        """ Update all statics on a step.  """
        # Update reward
        self._episodic.rewards += rewards
        self._episodic.episode[dones] += 1
        self._episodic.steps[np.bitwise_not(dones)] += 1

        self._episodic.steps[dones] = 0
        self._episodic.rewards[dones] = 0
        # TODO Store done episodes


class Episode:
    def __init__(self, instances):
        self.episode = np.zeros(instances)
        self.steps = np.zeros(instances)
        self.rewards = np.zeros(instances)


class Log:
    def __init__(self, save_dir):
        pass

    def write(self):
        pass

    def read(self):
        pass
