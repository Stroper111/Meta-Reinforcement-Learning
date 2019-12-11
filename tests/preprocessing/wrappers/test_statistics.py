import unittest
import numpy as np
import os
import glob

from unittest.mock import Mock

from core.preprocessing.wrappers import Statistics


class TestStatistics(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = Mock()
        cls.env.setup = dict(coinrun=2, bigfish=2)
        cls.env.instances = sum(cls.env.setup.values())
        cls.current_directory = os.path.dirname(os.path.abspath(__file__))
        cls.full_path_directory = os.path.join(cls.current_directory, "temp")

        # Create fake interaction
        cls.step_counter = 1
        cls.wrapper = Statistics(cls.env, cls.full_path_directory)

    @classmethod
    def tearDownClass(cls) -> None:
        for file in glob.glob(os.path.join(cls.full_path_directory, "*.txt")):
            os.remove(file)

        if os.path.exists(cls.full_path_directory):
            os.removedirs(cls.full_path_directory)


    def _fake_reward(self):
        """ The reward is equal to the id instance of the game.  """
        return np.arange(self.env.instances)

    def _fake_dones(self):
        """ Every game will terminate 10 games later than the next one.  """
        dones = []
        for id in range(1, self.env.instances + 1):
            if (self.step_counter % (id * 10) == 0):
                dones.append(True)
            else:
                dones.append(False)
        self.step_counter += 1
        return np.array(dones, dtype=np.bool)

    def test_print_statistics(self):
        """ Test if the function returns as expected.  """
        self.wrapper.reset()
        stats_continuous, stats_episodic = self.wrapper.summary(stats=['mean'])
        mean = stats_continuous['mean']
        self.assertEqual((3, self.env.instances), stats_episodic.shape)
        self.assertEqual(True, np.array_equal(np.zeros_like(mean), mean))

    def test_update_statistics(self):
        self.wrapper._step_update(rewards=self._fake_reward(), dones=self._fake_dones())
        stats_episodic, stats_continuous = self.wrapper.summary(stats=['mean'])
        self.assertEqual(True, np.array_equal([0, 0.5, 1, 1.5], stats_continuous['mean']))

        for _ in range(1, self.wrapper.continuous_history_size + 1):
            self.wrapper._step_update(rewards=self._fake_reward(), dones=self._fake_dones())
        stats_episodic, stats_continuous = self.wrapper.summary(stats=['mean'])

        self.assertEqual(True, np.array_equal(np.arange(self.env.instances), stats_continuous['mean']))
        self.assertEqual(True, np.array_equal([3, 1, 1, 0], stats_episodic['episode']), "Wrong episode numbers")
        self.assertEqual(True, np.array_equal([0, 11, 2, 93], stats_episodic['rewards']), "Wrong reward numbers")
        self.assertEqual(True, np.array_equal([1, 11, 1, 31], stats_episodic['steps']), "Wrong steps number")
