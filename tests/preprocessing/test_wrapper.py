import unittest
import numpy as np
import os

from core import MultiEnv
from core.preprocessing.wrappers import *


class TestMultiEnvWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        setup_one = dict(coinrun=1)
        setup_two = dict(coinrun=6)
        setup_three = dict(coinrun=2, bigfish=2, chaser=2)
        cls.setups = dict(env_one=setup_one, env_two=setup_two, env_three=setup_three)
        cls.keys = ['env_one', 'env_two', 'env_three']

        cls.current_directory = os.path.dirname(os.path.abspath(__file__))
        cls.full_path_directory = os.path.join(cls.current_directory, "temp")

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.full_path_directory):
            os.removedirs(cls.full_path_directory)

    def setUp(self) -> None:
        for key, setup in self.setups.items():
            setattr(self, key, MultiEnv(setup))

    def test_rgb2gray_wrapper(self):
        for key in self.keys:
            env = RGB2Gray(getattr(self, key))

            images = env.reset()['rgb']
            self.assertEqual((sum(self.setups[key].values()), 64, 64), images.shape, "Shape mismatch")
            self.assertEqual(True, np.max(images) <= 255, "Images can have bigger values than 255")
            self.assertEqual(True, np.min(images) >= 0, "Image values can become negative. ")

    def test_frame_stack_wrapper(self):
        for key in self.keys:
            for stack in [1, 4]:
                env = FrameStack(getattr(self, key), stack=stack)

                images = env.reset()['rgb']
                self.assertEqual((sum(self.setups[key].values())), len(images), "No Lazy frames?")

                images = np.array(images)
                self.assertEqual((sum(self.setups[key].values()), stack, 64, 64, 3), images.shape, "Shape mismatch")
                self.assertEqual(True, np.max(images) <= 255, "Images can have bigger values than 255")
                self.assertEqual(True, np.min(images) >= 0, "Image values can become negative. ")

    def test_statistics_unique(self):
        for key in self.keys:
            env = StatisticsUnique(getattr(self, key), save_dir=self.full_path_directory)

            images = env.reset()['rgb']
            instances = sum(self.setups[key].values())
            self.assertEqual((instances, 64, 64, 3), images.shape, "Shape mismatch")
            self.assertEqual(True, np.max(images) <= 255, "Images can have bigger values than 255")
            self.assertEqual(True, np.min(images) >= 0, "Image values can become negative. ")

            stats = env.summary(stats=['mean'])
            self.assertEqual(True, np.array_equal(np.zeros(instances), stats['episode']),
                             "Wrong episode numbers")
            self.assertEqual(True, np.array_equal(np.zeros(instances), stats['rewards']),
                             "Wrong reward  numbers")
            self.assertEqual(True, np.array_equal(np.zeros(instances), stats['steps']),
                             "Wrong steps number")

    def test_rescaling(self):
        for key in self.keys:
            # Required for next wrapper
            env = RGB2Gray(getattr(self, key))
            env = RescalingGray(env, new_shape=(128, 128))

            images = env.reset()['rgb']
            instances = sum(self.setups[key].values())
            self.assertEqual((instances, 128, 128), images.shape, "Shape mismatch")

    def test_motion_tracer(self):
        for key in self.keys:
            # Required for next wrapper
            env = MotionTracer(getattr(self, key))

            images = env.reset()['rgb']
            instances = sum(self.setups[key].values())
            self.assertEqual((instances, 64, 64, 3, 2), images.shape, "Shape mismatch")

    def test_motion_tracer_gray(self):
        for key in self.keys:
            # Required for next wrapper
            env = RGB2Gray(getattr(self, key))
            env = MotionTracer(env)

            images = env.reset()['rgb']
            instances = sum(self.setups[key].values())
            self.assertEqual((instances, 64, 64, 2), images.shape, "Shape mismatch")

    def test_motion_tracer_rescale_gray(self):
        for key in self.keys:
            # Required for next wrapper
            env = RGB2Gray(getattr(self, key))
            env = RescalingGray(env, new_shape=(128, 128))
            env = MotionTracer(env)

            images = env.reset()['rgb']
            instances = sum(self.setups[key].values())
            self.assertEqual((instances, 128, 128, 2), images.shape, "Shape mismatch")


if __name__ == '__main__':
    unittest.main()
