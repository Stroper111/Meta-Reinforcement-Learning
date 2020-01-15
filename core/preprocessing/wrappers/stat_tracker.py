from abc import ABC

import numpy as np
import os

from collections import deque

from core.preprocessing.wrappers import BaseWrapper


class StatisticsUnique(BaseWrapper, ABC):
    """ Converter for MultiEnv generated images.  """

    def __init__(self, env, save_dir, history_size=30):
        super().__init__(env)
        self.save_dir = save_dir
        self.setup = env.setup
        self.instances = env.instances

        self._continuous_history_size = history_size
        self._save_paths = self._create_save_paths(save_dir, self.setup)

        self._continuous = Continuous(self.instances, history_size)
        self._episodic = Episode(self.instances, self._continuous, self._save_paths)

    @staticmethod
    def _create_save_paths(save_dir, setup):
        files = []
        for game, instances in setup.items():
            for each in range(instances):
                file = os.path.join(save_dir, f"logs_{game}_{'{:02d}'.format(each)}")
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
        return dict(**episodic, **continuous)

    def scheduler(self):
        """ Returns scheduler information.  """
        return sum(self._episodic.episode), self._continuous.total_steps

    def last_episode_info(self):
        """ Returns the episode, number of steps, reward and continuous mean of last episode.  """
        return self._episodic.last_episode_info

    def _step_update(self, rewards, dones):
        """ Update all statics on a step.  """
        self._episodic.update(rewards, dones)
        self._continuous.update(rewards, dones)


class Continuous:
    def __init__(self, instances, history_size):
        self._data = [deque([0], maxlen=history_size) for _ in range(instances)]
        self.instances = instances
        self.history_size = history_size
        self.total_steps = 0

    def summary(self, stats):
        data = {stat: [getattr(np, stat)(game) for game in self._data] for stat in stats}
        data['total_steps'] = [self.total_steps * self.instances]
        return data

    def update(self, rewards, dones):
        """ Update on every step.  """
        self.total_steps += 1

    def update_score(self, idx, reward):
        """  Updated at the end of the game.  """
        self._data[idx].append(reward)

    def stat_per_game(self, idx, stat):
        return getattr(np, stat)(self._data[idx])


class Episode:
    def __init__(self, instances, continuous, save_paths):
        self.episode = np.zeros(instances, dtype=np.int)
        self.steps = np.zeros(instances, dtype=np.int)
        self.rewards = np.zeros(instances, dtype=np.float)
        self.continuous = continuous
        self.save_paths = save_paths
        self.last_episode_info = dict(episode=0, steps=0, reward=0, mean=0)

    def summary(self):
        return dict(episode=self.episode, steps=self.steps, rewards=self.rewards)

    def update(self, rewards, dones):
        self.rewards += rewards
        self.steps += 1

        # Write completed episodes to a file
        for idx in np.where(dones)[0]:
            self.continuous.update_score(idx, self.rewards[idx])
            mean = self.continuous.stat_per_game(idx, stat='mean')

            self.write(idx, mean)
            self.steps[idx] = 0
            self.rewards[idx] = 0
            self.episode[idx] += 1

    def write(self, idx, mean):
        with open(f"{self.save_paths[idx]}.txt", mode='a', buffering=1) as file:
            msg = "{episode:9,d}\t{steps:8,d}\t{reward:6,.1f}\t{mean:9,.2f}\n"
            msg_info = dict(episode=self.episode[idx], steps=self.steps[idx], reward=self.rewards[idx], mean=mean)
            msg = msg.format(**msg_info)
            file.write(msg)
        self.last_episode_info = msg_info
