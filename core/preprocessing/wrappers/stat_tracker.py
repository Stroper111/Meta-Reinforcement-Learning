import numpy as np
import os

from collections import deque

from core.preprocessing.wrappers import BaseWrapper


class StatisticsUnique(BaseWrapper):
    """ Converter for MultiEnv generated images.  """

    def __init__(self, env, save_dir, history_size=30):
        super().__init__(env)
        self.save_dir = save_dir
        self.setup = env.setup
        self.instances = env.instances

        self.continuous_history_size = history_size
        self.save_paths = self._save_paths(save_dir, self.setup)

        self._continuous = Continuous(self.instances, history_size)
        self._episodic = Episode(self.instances, self._continuous, self.save_paths)

    def _save_paths(self, save_dir, setup):
        files = []
        for game, instances in setup.items():
            for each in range(instances):
                file = os.path.join(save_dir, f"logs_{game}_{each}")
                files.append(file)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return files

    def step(self, action):
        images, rewards, dones, infos = self.env.step(action)
        self._step_update(rewards, dones)
        return images, rewards, dones, infos

    def summary(self, stats: list = None):
        """ Print the information of the last statistics, stats have to be valid numpy functions.  """
        episodic = self._episodic.summary()
        continuous = self._continuous.summary(stats)
        return episodic, continuous

    def scheduler(self):
        return dict(episode=sum(self._episodic.episode), steps=self._continuous.total_steps)

    def _step_update(self, rewards, dones):
        """ Update all statics on a step.  """
        self._episodic.update(rewards, dones)
        self._continuous.update(rewards, dones)


class Continuous:
    def __init__(self, instances, history_size):
        self._data = [deque([0], maxlen=history_size) for _ in range(instances)]
        self.total_steps = 0

    def summary(self, stats):
        data = {stat: [getattr(np, stat)(game) for game in self._data] for stat in stats}
        data['total_steps'] = self.total_steps
        return data

    def update(self, rewards, dones):
        [game.append(data) for game, data in zip(self._data, rewards)]
        # This is the same for all games
        self.total_steps += 1

    def stat_per_game(self, idx, stat):
        return getattr(np, stat)(self._data[idx])


class Episode:
    def __init__(self, instances, continuous,  save_paths):
        self.episode = np.zeros(instances, dtype=np.int)
        self.steps = np.zeros(instances, dtype=np.int)
        self.rewards = np.zeros(instances, dtype=np.float)
        self.continuous = continuous
        self.save_paths = save_paths

    def summary(self):
        return dict(episode=self.episode, steps=self.steps, rewards=self.rewards)

    def update(self, rewards, dones):
        self.rewards += rewards
        self.steps += 1

        # Write completed episodes to a file
        for idx in np.where(dones)[0]:
            mean = self.continuous.stat_per_game(idx, stat='mean')
            self.write(idx, mean)
            self.steps[idx] = 0
            self.rewards[idx] = 0
            self.episode[idx] += 1

    def write(self, idx, mean):
        with open(f"{self.save_paths[idx]}{'%02d' % idx}.txt", mode='a', buffering=1) as file:
            msg = "{episode:9,d}\t{steps:12,d}\t{reward:6,.1f}\t{mean:9,.2f}\n"
            msg = msg.format(episode=self.episode[idx], steps=self.steps[idx], reward=self.rewards[idx], mean=mean)
            file.write(msg)
