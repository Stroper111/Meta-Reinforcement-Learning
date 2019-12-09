import unittest
import numpy as np

from core import MultiEnv
from core.preprocessing.wrappers import RGB2Gray


class TestMultiEnvWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        setup_one = dict(coinrun=1)
        setup_two = dict(coinrun=6)
        setup_three = dict(coinrun=2, bigfish=2, chaser=2)
        cls.setups = dict(env_one=setup_one, env_two=setup_two, env_three=setup_three)
        cls.keys = ['env_one', 'env_two', 'env_three']

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


if __name__ == '__main__':
    unittest.main()
