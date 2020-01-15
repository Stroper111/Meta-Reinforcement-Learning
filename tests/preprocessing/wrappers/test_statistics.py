import unittest
import numpy as np
import os
import glob

from unittest.mock import Mock

from core.preprocessing.wrappers import StatisticsUnique


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
        cls.wrapper = StatisticsUnique(cls.env, cls.full_path_directory)

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
        stats = self.wrapper.summary(stats=['mean'])
        self.assertEqual(['episode', 'steps', 'rewards', 'mean', 'total_steps'], list(stats.keys()))
        self.assertEqual(True, np.array_equal(np.zeros_like(stats['mean']), stats['mean']))

    def test_update_statistics(self):
        self.wrapper._step_update(rewards=self._fake_reward(), dones=self._fake_dones())

        for _ in range(1, self.wrapper._continuous_history_size + 1):
            self.wrapper._step_update(rewards=self._fake_reward(), dones=self._fake_dones())
        stats = self.wrapper.summary(stats=['mean'])
        total_steps = (self.wrapper._continuous_history_size + 1) * self.env.instances

        self.assertEqual(True, np.array_equal([3, 1, 1, 0], stats['episode']), "Wrong episode numbers")
        self.assertEqual(True, np.array_equal([0, 11, 2, 93], stats['rewards']), "Wrong reward numbers")
        self.assertEqual(True, np.array_equal([1, 11, 1, 31], stats['steps']), "Wrong steps number")
        self.assertEqual([total_steps], stats['total_steps'], "Wrong step count")
        self.assertEqual(True, np.array_equal([0, 10, 30, 0], stats['mean']))

        # Test for saving, if no clean up was required, saving is not working properly.
        self._clean_up()

    def _clean_up(self):
        remove = False
        for file in glob.glob(os.path.join(self.full_path_directory, "*.txt")):
            os.remove(file)
            remove = True

        if os.path.exists(self.full_path_directory):
            os.removedirs(self.full_path_directory)

        self.assertEqual(True, remove, "No logs removed, test saving...")