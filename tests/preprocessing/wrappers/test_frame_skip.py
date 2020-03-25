import unittest
import pickle
import numpy as np
import os

from unittest.mock import Mock

from core import MultiEnv
from core.preprocessing.wrappers.frame_skip import FrameSkip


class TestFrameStack(unittest.TestCase):

    @classmethod
    def setUp(self) -> None:
        self.venv = MultiEnv({'Pong-v0': 1})

        self.instances = self.venv.instances
        self.actions = np.zeros((self.instances,), dtype=np.uint8)
        self.frame_counter = 0
        self.frame_skip = 4

        self.wrapper = FrameSkip(self.venv, self.frame_skip)

    def test_skipping_on_reset(self):
        for times in range(5):
            img = self.wrapper.reset()
            self.assertEqual(self.frame_skip, self.wrapper._frame_counter, "Frameskip is not performed on start.")

            for _ in range(times):
                self.wrapper.step(self.actions)

    def test_skipping_on_steps(self):
        """ Checks the skipping of frames.  """
        self.wrapper.reset()
        frame_counter = 1
        for _ in range(10):
            img, reward, done, info = self.wrapper.step(actions=np.zeros((self.instances,), dtype=np.uint8))

            frame_counter += 1
            frame_count = frame_counter * self.frame_skip

            self.assertEqual(frame_count, self.wrapper._frame_counter, "Make sure that the frame skip is correct")
            self.assertEqual((self.instances,), reward.shape, "Different reward shape than expected.")
            self.assertEqual((self.instances,), done.shape, "Different done shape than expected.")
            self.assertEqual((self.instances,), info.shape, "Different info shape than expected.")

    def test_done(self):
        """ Checks the skipping of frames on reset.  """
        self.wrapper.env.step = self._step
        img = self.wrapper.reset()
        self.assertEqual(4, self.wrapper._frame_counter)

    def _step(self, action):
        img = np.ones(self.instances) * self.frame_counter
        reward = np.ones(self.instances) * self.frame_counter
        done = np.zeros(self.instances)
        info = np.array([dict() for _ in range(self.instances)])

        self.frame_counter += 1
        return img, reward, done, info


if __name__ == '__main__':
    unittest.main()
