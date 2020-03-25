import numpy as np
import unittest

from core.memory import BaseMemoryNumpy


class TestReplayMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.size = 100
        self.shape = (64, 64, 3)
        self.action_space = 5

    def test_shapes(self):
        for shape in [(4,), (1,), (64, 64, 3), (210, 160, 3)]:
            memory = BaseMemoryNumpy(size=self.size, shape=shape, action_space=self.action_space, stacked_frames=False)
            self.assertEqual((self.size, *shape), memory.states.shape, "State shape mismatch.")

            memory = BaseMemoryNumpy(size=self.size, shape=shape, action_space=self.action_space, stacked_frames=True)
            self.assertIsInstance(memory.states, list, "Stacked frames not handle correctly.")
            self.assertEqual(self.size, len(memory.states), "State shape mismatch on stacked frames.")

    def test_pointer_filling(self):
        memory = BaseMemoryNumpy(size=self.size, shape=self.shape, action_space=self.action_space, stacked_frames=False)
        self._add_data(memory, 51)
        self.assertEqual(51, memory.pointer, "Wrong memory completeness indication.")
        self.assertEqual(False, memory.filled, "Wrong indication of filled memory.")
        self.assertEqual(51, len(memory), "Memory length not consistent.")

    def test_memory_overflow(self):
        memory = BaseMemoryNumpy(size=self.size, shape=self.shape, action_space=self.action_space, stacked_frames=False)
        self._add_data(memory, self.size + 1)
        self.assertEqual(1, memory.pointer, "Wrong memory completeness indication after overflow.")
        self.assertEqual(True, memory.filled, "Wrong indication of filled memory after overflow.")
        self.assertEqual(self.size, len(memory), "Memory length not consistent after overflow.")

    def test_get_batch(self):
        memory = BaseMemoryNumpy(size=self.size, shape=self.shape, action_space=self.action_space, stacked_frames=False)
        self._add_data(memory, self.size + 1)

        batch_size = 32
        states, actions, rewards, done, states_next = memory.get_batch(batch_size)

        self.assertEqual((batch_size, *self.shape), states.shape, "Batch states shape mismatch.")
        self.assertEqual((batch_size,), actions.shape, "Batch actions shape mismatch.")
        self.assertEqual((batch_size,), rewards.shape, "Batch rewards shape mismatch.")
        self.assertEqual((batch_size,), done.shape, "Batch done shape mismatch.")
        self.assertEqual((batch_size, *self.shape), states_next.shape, "Batch states next shape mismatch.")

    def _add_data(self, memory, number):
        """ Helper to create random data.  """
        for iteration in range(number):
            state = np.random.randint(0, 255, self.shape)
            action = iteration % self.action_space
            reward = iteration
            done = 1 if iteration % 25 == 24 else 0
            next_state = np.random.randint(0, 255, self.shape)
            memory.add(state, action, reward, done, next_state)
