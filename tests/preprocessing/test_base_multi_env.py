import unittest
import numpy as np
import os

from core.tools import MultiEnv


# TODO Rebuild
class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup = dict(coinrun=12)
        cls.instance = sum(cls.setup.values())
        cls.env = MultiEnv(cls.setup)
        cls.current_directory = os.path.dirname(os.path.abspath(__file__))
        cls.full_path_directory = os.path.join(cls.current_directory, "temp")

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.full_path_directory):
            os.removedirs(cls.full_path_directory)

    def test_default(self):
        env = BasePreProcessingMultiEnv(self.env, save_dir=self.full_path_directory).env
        images = env.reset()
        self.assertEqual(self.instance, len(images['rgb']))

        images_stack = np.array(images['rgb'])
        self.assertEqual((self.instance, 4, 64, 64), images_stack.shape)

        images, *_ = env.step(np.zeros(self.instance))
        images_stack = np.array(images['rgb'])
        self.assertEqual((self.instance, 4, 64, 64), images_stack.shape)

    def test_rgb2gray(self):
        env = BasePreProcessingMultiEnv(env=self.env,
                                        rgb2gray=True,
                                        frame_stack=None,
                                        statistics=False).env

        images = env.reset()
        self.assertEqual((self.instance, 64, 64), images['rgb'].shape)

        images, *_ = env.step(np.zeros(self.instance))
        self.assertEqual((self.instance, 64, 64), images['rgb'].shape)

    def test_stacking_4(self):
        env = BasePreProcessingMultiEnv(env=self.env,
                                        rgb2gray=False,
                                        frame_stack=4,
                                        statistics=False).env
        images = env.reset()
        images = np.array(images['rgb'])
        self.assertEqual((self.env.instances, 4, 64, 64, 3), images.shape, "Different shape than expected")
        self.assertEqual(True, np.array_equal(images[0][0], images[0][3]), "Images not copied")

    def test_stacking_none(self):
        env = BasePreProcessingMultiEnv(env=self.env,
                                        rgb2gray=False,
                                        frame_stack=False,
                                        statistics=False).env
        images = env.reset()
        self.assertEqual((self.env.instances, 64, 64, 3), images['rgb'].shape, "Different shape than expected")



if __name__ == '__main__':
    unittest.main()
