import unittest

from core.preprocessing.wrappers import GymWrapper


class TestGymWrapper(unittest.TestCase):

    def test_single_setup(self):
        for each in ["Taxi-v3", "CartPole-v0", "Breakout-v0", "MsPacman-v0"]:
            env = GymWrapper(each, 1)
            env.reset()
            env.step([0])
            env.close()

    def test_single_setup_duration_test(self):
        env = GymWrapper("CartPole-v0")
        env.reset()
        has_reset = False

        for _ in range(250):
            img, reward, done, info = env.step([0])
            has_reset = has_reset or done

        self.assertEqual(True, has_reset, msg="No reset has occurred.")

    def test_multi_instance(self):
        for game, instance in [("Taxi-v3", 2), ("MsPacman-v0", 2)]:
            env = GymWrapper(game, instance)
            env.reset()

            for _ in range(100):
                actions = [action_space.sample() for action_space in env.action_space]
                env.step(actions)

    def test_multiple_instances_shape(self):
        instances = 5
        env = GymWrapper("MsPacman-v0", instances)
        env.reset()

        actions = [action_space.sample() for action_space in env.action_space]
        img, reward, done, info = env.step(actions)

        self.assertEqual((instances, 210, 160, 3), img.shape, msg="Shape mismatch img.")
        self.assertEqual((instances,), reward.shape, msg="Shape mismatch reward.")
        self.assertEqual((instances,), done.shape, msg="Shape mismatch done.")
        self.assertEqual((instances,), info.shape, msg="Shape mismatch info.")
