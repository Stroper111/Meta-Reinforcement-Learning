import numpy as np
import unittest

from core.memory import ReplayMemory


class TestReplayMemory(unittest.TestCase):

    def setUp(self) -> None:
        self.size = 100
        self.shape = (64, 64, 3)
        self.action_space = 5
        self.alpha = 0.1
        self.gamma = 0.97

        self.memory = ReplayMemory(size=self.size, shape=self.shape, action_space=self.action_space,
                                   alpha=self.alpha, gamma=self.gamma)

    def test_steps(self):
        self._add_data(51)
        self.assertEqual(51, self.memory.pointer, "Wrong memory completeness indication.")
        self.assertEqual(False, self.memory.filled, "Wrong indication of filled memory.")
        self.assertEqual(51, len(self.memory), "Memory length not consistent.")

    def test_overflow(self):
        self._add_data(self.size)
        self.memory.refill_memory()
        self._add_data(1)

        self.assertEqual(True, self.memory.filled, "Memory filled not set correctly on overflow.")
        self.assertEqual(1, self.memory.pointer, "Memory in use not reset correctly on overflow.")

    def test_update(self):
        """  This only tests the execution of the function.  """
        self._add_data(self.size - 1)
        self.memory.update()
        # Expected output is 1 * 1.97**steps for 1 q value, and 1 if episode is done. (every 25 steps)
        # self.memory.q_values

    def _add_data(self, number):
        """ Helper to create random data.  """
        for iteration in range(number):
            state = np.random.randint(0, 255, self.shape)
            q_values = np.zeros(self.action_space)
            action = iteration % self.action_space
            reward = 1
            end_episode = 1 if iteration % 25 == 0 else 0
            self.memory.add(state, q_values, action, reward, end_episode)


if __name__ == '__main__':
    unittest.main()
