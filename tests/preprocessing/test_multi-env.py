import unittest
import numpy as np

from core import MultiEnv


class TestMultiEnv(unittest.TestCase):
    def setUp(self):
        self.setup = dict(coinrun=2, bigfish=2, chaser=2)
        self.env_number = sum(self.setup.values())

        self.env = MultiEnv(self.setup)
        self.env_action_space = self.env.action_space['coinrun'].n

    def tearDown(self) -> None:
        self.env.close()

    def test_wrong_setups(self):
        with self.assertRaises(AssertionError):
            env = MultiEnv({"MsPacman-v0": 1})

        with self.assertRaises(AssertionError):
            env = MultiEnv(dict(coinrun=6.5))

        with self.assertRaises(AssertionError):
            env = MultiEnv(dict(coinrun="6"))

    def test_images_on_reset(self):
        images = self.env.reset()
        self.assertEqual((self.env_number, 64, 64, 3), images['rgb'].shape)

    def test_step(self):
        self.env.reset()
        for _ in range(500):
            actions = np.random.randint(0, self.env_action_space, self.env_number)
            self.env.step(actions)

    def test_step_output(self):
        self.env.reset()
        actions = np.random.randint(0, self.env_action_space, self.env_number)
        images, rewards, dones, infos = self.env.step(actions)

        self.assertEqual((self.env_number, 64, 64, 3), images['rgb'].shape, "In step image changed shape.")
        self.assertEqual(self.env_number, len(rewards), "Got different amount of rewards than environments.")
        self.assertEqual(self.env_number, len(dones), "Got different amount of dones than environments.")
        self.assertEqual(self.env_number, len(infos), "Got different amount of infos than environments.")

    def test_render(self):
        self.env.reset()
        self.env.render()


if __name__ == '__main__':
    unittest.main()
