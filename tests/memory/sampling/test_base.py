import numpy as np
import unittest

from unittest.mock import Mock

from core.memory import BaseSampling


class TestSampling(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 32
        self.memory_size = 512

        self.replay_memory = Mock()
        self.replay_memory.__len__ = Mock(return_value=self.memory_size)
        self.replay_memory.get_batch.return_value = self._get_batch(np.random.rand(self.batch_size))

        self.sampling = BaseSampling(replay_memory=self.replay_memory, batch_size=self.batch_size)

    def test_generator(self):
        """  Test if there are exactly the correct number of batches to reproduce the whole memory.  """
        counter = 0
        for counter, batch in enumerate(self.sampling):
            pass
        self.assertEqual(self.memory_size // self.batch_size, counter + 1, "Number of batches not calculated correctly")

    def _get_batch(self, idx):
        """  Return fake observations and q_values.  """
        return np.random.rand(len(idx), 64, 64, 3), np.random.rand(len(idx), 6)

if __name__ == '__main__':
    unittest.main()
