

import unittest
import pickle
import numpy as np
import time
import os

from unittest.mock import Mock

from core import MultiEnv
from core.preprocessing.wrappers import RGB2Gray, FrameStack


class TestWrappers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        current_directory = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_directory, "_original_images.pkl"), "rb") as file:
            cls.images = pickle.load(file)

        cls.env = Mock()
        cls.env.instances = 12
        cls.env.reset.return_value = cls.images
        cls.env.step.return_value = (cls.images, None, None, None)
        cls.wrapper = FrameStack(RGB2Gray(cls.env), 4)

    def test_reset(self):
        images = self.wrapper.reset()
        print(images['rgb'])
        print(np.array(images['rgb']).shape)
        self.assertEqual(True, False)
