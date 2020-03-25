import time
import numpy as np

from typing import List
from collections import deque

from core.preprocessing.wrappers import BaseWrapper


class EpisodeStatistics(BaseWrapper):
    """ Keeps track of episode statistics.  """

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.env = env
        self.instances = env.instances

        self.total_episodes = 0
        self.total_steps = 0

        self.start_time = time.time()
        self.game_time: list = [time.time() for _ in range(self.instances)]

        self.episode_return: np.ndarray = np.zeros((self.instances,), np.float32)
        self.episode_length: np.ndarray = np.zeros((self.instances,), np.int)

        self.return_queue: List[deque] = [deque(maxlen=deque_size) for _ in range(self.instances)]
        self.length_queue: List[deque] = [deque(maxlen=deque_size) for _ in range(self.instances)]

    def step(self, actions):
        """ Increment all counter and update environment whenever the game is done.  """
        img, reward, done, info = self.env.step(actions)

        self.episode_return += reward
        self.episode_length += 1
        self.total_steps += self.instances

        for idx in np.where(done)[0]:
            info[idx]['episode'] = {'r': self.episode_return[idx],
                                    'l': self.episode_length[idx],
                                    't': round(time.time() - self.game_time[idx], 6)}

            self.return_queue[idx].append(self.episode_return[idx])
            self.length_queue[idx].append(self.episode_length[idx])

            self.total_episodes += 1
            self.episode_return[idx] = 0.0
            self.episode_length[idx] = 0
            self.game_time[idx] = time.time()

        return img, reward, done, info

    def reset(self):
        """ reset all counters.  """
        observation = self.env.reset()

        self.game_time = [time.time() for _ in range(self.instances)]
        self.episode_return = np.zeros((self.instances,), np.float32)
        self.episode_length = np.zeros((self.instances,), np.int)
        self.total_episodes = 0
        self.total_steps = 0

        return observation

    def statistics(self):
        """ Print out the statistics in a human readable format.  """
        print(f"\nSummary (elapsed time: {self._convert_time()}, episodes: {'{:6,d}'.format(self.total_episodes)})")
        print(f"\t{'%-15s' % 'game'}{''.join([('%15s' % key) * instance for key, instance in self.env.setup.items()])}")
        print(f"\t{'reward'.ljust(15)}{''.join(['{:15,.2f}'.format(score) for score in self.episode_return])}")
        print(f"\t{'steps'.ljust(15)}{''.join(['{:12,d}   '.format(steps) for steps in self.episode_length])}")

        for stat, values, func in [('Avg reward', self.return_queue, np.mean),
                                   ('Avg steps', self.length_queue, np.mean)]:
            print(f"\t{stat.ljust(15)}{''.join(['{:15,.2f}'.format(func(value)) for value in values])}")

    def summary(self) -> dict:
        """ Return a concise summary of overall game state.  """
        return dict(episode=self.total_episodes, steps=self.total_steps, time=time.time() - self.start_time)

    def _convert_time(self):
        """ Convert seconds to a human readable format.  """
        seconds = round(time.time() - self.start_time)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)

        if d:
            return '{d:02d} days {h:02d} hours {m:02d} min {s:02d} sec'.format(s=s, m=m, h=h, d=d)
        return '{h:02d} hours {m:02d} min {s:02d} sec'.format(s=s, m=m, h=h)
