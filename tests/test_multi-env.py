import unittest

from core import MultiEnv


class TestMultiEnv(unittest.TestCase):
    def setUp(self):
        self.setup = dict(coinrun=2, bigfish=2, chaser=2)
        self.env_number = sum(self.setup.values())
        self.env = MultiEnv(self.setup)

    def test_images_on_reset(self):
        images = self.env.reset()
        self.assertEqual((self.env_number, 64, 64, 3), images['rgb'].shape)

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
