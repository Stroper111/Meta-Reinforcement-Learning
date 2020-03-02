import numpy as np
import unittest
import procgen

from core.preprocessing.wrappers import UnpackRGB


class TestUnpackRGB(unittest.TestCase):

    def test_procgen_unpack(self):
        instances = 5
        for each in ["coinrun", "bigfish"]:
            env = procgen.ProcgenEnv(env_name=each, num_envs=instances)
            env = UnpackRGB(env=env)

            actions = np.array([env.action_space.sample() for _ in range(instances)])
            env.reset()
            img, reward, done, info = env.step(actions)

            self.assertEqual(True, np.ndarray == type(img), msg="Image is not unpacked.")
            self.assertEqual((instances, 64, 64, 3), img.shape, msg="Shape mismatch img.")
            self.assertEqual((instances,), reward.shape, msg="Shape mismatch reward.")
            self.assertEqual((instances,), done.shape, msg="Shape mismatch done.")
            self.assertEqual(instances, len(info), msg="Shape mismatch info.")

            env.close()
