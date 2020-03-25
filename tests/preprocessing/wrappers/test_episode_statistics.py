import unittest

import time
import numpy as np

from unittest.mock import Mock
from core.preprocessing.wrappers import EpisodeStatistics


class TestMultiEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.venv = Mock()
        cls.venv.setup = dict(pong=3, pacman=3)
        cls.venv.instances = sum(cls.venv.setup.values())

        cls.instances = cls.venv.instances
        cls.venv.reset.return_value = np.zeros((cls.instances,))

        cls.wrapper = EpisodeStatistics(cls.venv)
        cls.actions = np.zeros((cls.venv.instances,))

    def setUp(self) -> None:
        self.venv.step = self._return_normal

    def test_step(self):
        self.wrapper.reset()

        for step in range(1, 10):
            *_, info = self.wrapper.step(self.actions)

            for idx, each in enumerate(info):
                self.assertEqual(step * 2, self.wrapper.episode_return[idx], 'Incorrect reward score')
                self.assertEqual(step, self.wrapper.episode_length[idx], 'Incorrect number of steps')

        self.assertEqual(9 * self.instances, self.wrapper.total_steps, "Incorrect total number of steps.")

    def test_reset(self):
        self.wrapper.reset()

        for _ in range(10):
            *_, info = self.wrapper.step(self.actions)

        end_time = time.time() - self.wrapper.game_time[0]
        self.wrapper.reset()

        *_, info = self.wrapper.step(self.actions)

        for idx, each in enumerate(info):
            self.assertEqual(2, self.wrapper.episode_return[idx], 'Incorrect reward score after reset')
            self.assertEqual(1, self.wrapper.episode_length[idx], 'Incorrect number of steps')
            self.assertEqual(True, self.wrapper.game_time[idx] > end_time, 'Time is not reset properly.')

    def test_statistics(self):
        """ This is a visual test.  """
        self.wrapper.reset()
        self.wrapper.statistics()

        for _ in range(100):
            self.wrapper.step(self.actions)

        time.sleep(2)
        self.wrapper.statistics()

    def test_done(self):
        self.wrapper.reset()
        # Play the game normally for 9 steps
        for _ in range(1, 10):
            *_, info = self.wrapper.step(self.actions)

        # Send a done signal to the first game.
        self.venv.step = self._return_done
        *_, info = self.wrapper.step(self.actions)
        self.venv.step = self._return_normal

        self.assertEqual(20, info[0]['episode']['r'], "Incorrect dict value reward after done.")
        self.assertEqual(10, info[0]['episode']['l'], "Incorrect dict value step after done.")
        self.assertEqual(1, self.wrapper.total_episodes, "Incorrect number of done episodes.")

        # Play another 10 steps.
        info = None
        for _ in range(10):
            *_, info = self.wrapper.step(self.actions)

        for idx, each in enumerate(info):
            if not idx:
                self.assertEqual(10 * 2, self.wrapper.episode_return[idx], 'Incorrect reward score after done')
                self.assertEqual(10, self.wrapper.episode_length[idx], 'Incorrect number of steps after done')
            else:
                self.assertEqual(20 * 2, self.wrapper.episode_return[idx], 'Incorrect reward score after not done')
                self.assertEqual(20, self.wrapper.episode_length[idx], 'Incorrect number of steps after not done')

        self.wrapper.statistics()

    def _return_done(self, *_):
        return np.ones((self.instances,)), \
               np.ones((self.instances,)) * 2, \
               np.array([1] + [0] * (self.instances - 1), dtype=np.bool), \
               np.array([dict() for _ in range(self.instances)])

    def _return_normal(self, *_):
        return np.zeros((self.instances,)), \
               np.ones((self.instances,)) * 2, \
               np.zeros((self.instances,), dtype=np.bool), \
               np.array([dict() for _ in range(self.instances)])
