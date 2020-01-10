import unittest
import os
import glob
import time
import numpy as np

from core import MultiEnv
from core.tools import Scheduler


class TestMultiEnv(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.current_directory = os.path.dirname(os.path.abspath(__file__))
        cls.full_path_directory = os.path.join(cls.current_directory, "temp_statistics")

    @classmethod
    def tearDownClass(cls) -> None:
        for file in glob.glob(os.path.join(cls.full_path_directory, "*.txt")):
            os.remove(file)

        if os.path.exists(cls.full_path_directory):
            os.removedirs(cls.full_path_directory)

    def setUp(self):
        self.setup = dict(coinrun=2, bigfish=2, chaser=2)
        self.instances = sum(self.setup.values())

        self.env = MultiEnv(self.setup)

    def test_setup(self):
        kwargs = dict()
        counter = 0
        for each in Scheduler(self.env, **kwargs):
            counter += 1
            if counter > 10:
                break

    def test_time_limit(self):
        kwargs = dict(time_limit=2, time_update=1)
        start_time = time.time()
        for each in Scheduler(self.env, **kwargs):
            if (time.time() - start_time) >= 2.5:
                self.fail("Not terminated within expected time.")

    def test_time_limit_update(self):
        """ Checks that the condition works if time update is higher than limit. """
        kwargs = dict(time_limit=2, time_update=10)
        start_time = time.time()
        for each in Scheduler(self.env, **kwargs):
            if (time.time() - start_time) >= (kwargs['time_limit'] + 0.2):
                self.fail("Not terminated within expected time.")

    def test_step_limit(self):
        kwargs = dict(step_limit=100, step_update=50)
        for env, update, episode, steps in Scheduler(self.env, **kwargs):
            env.step(np.zeros(self.instances))
            if steps > kwargs['step_limit']:
                self.fail("Not terminated within expected steps.")

    def test_step_limit_update(self):
        """ Checks that the condition works if step update is higher than limit. """
        kwargs = dict(step_limit=100, step_update=150)
        for env, update, episode, steps in Scheduler(self.env, **kwargs):
            env.step(np.zeros(self.instances))
            if steps > kwargs['step_limit']:
                self.fail("Not terminated within expected steps.")

    def test_episode_limit(self):
        kwargs = dict(episode_limit=5, episode_update=3)
        for env, update, episode, steps in Scheduler(self.env, **kwargs):
            _, _, dones, _ = env.step(np.zeros(self.instances))
            if episode > kwargs['episode_limit']:
                self.fail("Not terminated within expected epsiodes.")

    def test_episode_limit_update(self):
        """ Checks that the condition works if step update is higher than limit. """
        kwargs = dict(episode_limit=5, episode_update=10)
        for env, update, episode, steps in Scheduler(self.env, **kwargs):
            _, _, dones, _ = env.step(np.zeros(self.instances))
            if episode > kwargs['episode_limit']:
                self.fail("Not terminated within expected epsiodes.")

    def test_combinations(self):
        kwargs = dict(step_limit=500, episode_limit=10)
        for env, update, episode, steps in Scheduler(self.env, **kwargs):
            _, _, dones, _ = env.step(np.zeros(self.instances))
            if episode > kwargs['episode_limit']:
                self.fail("Not terminated within expected epsiodes.")

            if steps > kwargs['step_limit']:
                self.fail("Not terminated within expected steps.")
            steps += 1
