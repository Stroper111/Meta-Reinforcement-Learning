import unittest
import pickle
import numpy as np
import os

from unittest.mock import Mock

from core.preprocessing.wrappers.frame_stacker import FrameStack, LazyFrames


class TestFrameStack(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = Mock()
        cls.env.instances = 12
        cls.wrapper = FrameStack(cls.env, 4)

        current_directory = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current_directory, "data", "_original_images.pkl"), "rb") as file:
            cls.images = pickle.load(file)['rgb']

    def test_stacking(self):
        """ Checks that the results are LazyFrames and array output.  """
        self.wrapper._create_stacks(self.images)

        stacked_images = self.wrapper._get_images()
        [self.assertIsInstance(image, LazyFrames) for image in stacked_images]

        stacked_images = np.array(stacked_images)
        self.assertEqual((self.env.instances, 4, 64, 64, 3), stacked_images.shape, "Different shape than expected")
        self.assertEqual(True, np.array_equal(stacked_images[0][0], stacked_images[0][3]), "Images not copied")
