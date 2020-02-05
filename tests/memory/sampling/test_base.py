import numpy as np
import unittest

from unittest.mock import Mock

from core.memory import BaseSamplingMultiEnv


class TestSampling(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 32
        self.memory_size = 512
        self.shape = (64, 64, 3)
        self.action_space = 6

        self.model = Mock()
        self.model.predict.return_value = self._get_model_predict(self.batch_size)
        self.replay_memory = Mock()
        self.replay_memory.__len__ = Mock(return_value=self.memory_size)
        self.replay_memory.get_batch.return_value = self._get_batch(self.batch_size)

        self.sampling = BaseSamplingMultiEnv(replay_memory=self.replay_memory, model=self.model,
                                             alpha=0.1, gamma=0.95, batch_size=self.batch_size)

    def test_generator(self):
        """  Test if there are exactly the correct number of batches to reproduce the whole memory.  """
        counter = 0
        for counter, batch in enumerate(self.sampling):
            pass
        self.assertEqual(self.memory_size // self.batch_size, counter + 1, "Number of batches not calculated correctly")

    def test_create_model_input(self):
        model_input, model_output = self.sampling._create_model_input(*self._get_batch(self.batch_size))
        # This is required whenever there is no stacking
        model_input = model_input.transpose([0, 3, 1, 2])
        self.assertEqual((self.batch_size, *self.shape), model_input.shape, "Unequal dimensions of random batch")
        self.assertEqual((self.batch_size, self.action_space), model_output.shape, "Unequal dimensions of random batch")

    def _get_batch(self, batch_size):
        """  Return fake states, actions, rewards, done, states_next.  """
        return np.random.rand(batch_size, *self.shape), \
               np.random.randint(0, self.action_space, (batch_size,)), \
               np.random.randint(0, 1, (batch_size,)), \
               np.random.choice([0, 1], size=(batch_size,), p=[0.95, 0.05]), \
               np.random.rand(batch_size, *self.shape)

    def _get_model_predict(self, batch_size):
        """ Return fake q_values.  """
        return np.random.rand(batch_size, self.action_space)


if __name__ == '__main__':
    unittest.main()
