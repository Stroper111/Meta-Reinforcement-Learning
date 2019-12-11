import unittest
import numpy as np
import os
import pickle

from core.preprocessing import PreProcessingBase
from core.tools import MultiEnv


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup = dict(coinrun=12)
        cls.instance = sum(cls.setup.values())
        cls.env = MultiEnv(cls.setup)
        cls.current_directory = os.path.dirname(os.path.abspath(__file__))
        cls.full_path_directory = os.path.join(cls.current_directory, "temp")

    def test_rgb2gray(self):
        env = PreProcessingBase(env=self.env,
                                rgb2gray=True,
                                frame_stack=None,
                                statistics=False).env

        images = env.reset()
        self.assertEqual((self.instance, 64, 64), images['rgb'].shape)

        images, *_ = env.step(np.zeros(self.instance))
        self.assertEqual((self.instance, 64, 64), images['rgb'].shape)

    def test_stacking_4(self):
        env = PreProcessingBase(env=self.env,
                                rgb2gray=False,
                                frame_stack=4,
                                statistics=False).env
        images = env.reset()
        images = np.array(images['rgb'])
        self.assertEqual((self.env.instances, 4, 64, 64, 3), images.shape, "Different shape than expected")
        self.assertEqual(True, np.array_equal(images[0][0], images[0][3]), "Images not copied")

    def test_stacking_none(self):
        env = PreProcessingBase(env=self.env,
                                rgb2gray=False,
                                frame_stack=False,
                                statistics=False).env
        images = env.reset()
        self.assertEqual((self.env.instances, 64, 64, 3), images['rgb'].shape, "Different shape than expected")


if __name__ == '__main__':
    unittest.main()
