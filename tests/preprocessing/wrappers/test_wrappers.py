

import unittest
import pickle
import numpy as np
import time
import os

from unittest.mock import Mock

from core import MultiEnv
from core.preprocessing.wrappers import RGB2Gray, FrameStack
from core.preprocessing.wrappers.frame_stacker import LazyFrames


class TestWrappers(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        current_directory = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_directory, "_original_images.pkl"), "rb") as file:
            cls.images = pickle.load(file)

        cls.env = Mock()
        cls.env.instances = 12
        cls.env.action_space = 15
        cls.env.reset.return_value = cls.images
        cls.env.step.return_value = (cls.images, None, None, None)
        cls.wrapper = FrameStack(RGB2Gray(cls.env), 4)

    def test_reset(self):
        images = self.wrapper.reset()['rgb']
        images_array = np.array(images)
        [self.assertIsInstance(image, LazyFrames) for image in images]
        self.assertEqual((self.env.instances, 4, 64, 64), images_array.shape, "Different shape than expected")
        self.assertEqual(True, np.array_equal(images_array[0][0], images_array[0][-1]), "Images not copied")

    def test_step(self):
        self.wrapper.reset()
        actions = np.random.randint(0, self.env.action_space, self.env.instances)
        images, rewards, dones, infos = self.wrapper.step(actions)
        images_array = np.array(images['rgb'])

        [self.assertIsInstance(image, LazyFrames) for image in images['rgb']]
        self.assertEqual((self.env.instances, 4, 64, 64), images_array.shape, "In step image changed shape.")
