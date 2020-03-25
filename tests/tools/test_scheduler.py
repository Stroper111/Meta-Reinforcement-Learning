import unittest
import time
import numpy as np

from core import MultiEnv
from core.tools import Controller

# TODO Rebuild
class TestMultiEnv(unittest.TestCase):

    def setUp(self):
        self.setup = dict(coinrun=2, bigfish=2, chaser=2)
        self.instances = sum(self.setup.values())
        self.env = MultiEnv(self.setup)

    def test_setup(self):
        kwargs = dict()
        counter = 0
        for each in Controller(self.env, **kwargs):
            counter += 1
            if counter > 10:
                break

    def test_time_limit(self):
        kwargs = dict(time_limit=2, time_update=1)
        start_time = time.time()
        for each in Controller(self.env, **kwargs):
            if (time.time() - start_time) >= 2.5:
                self.fail("Not terminated within expected time.")

    def test_time_limit_update(self):
        """ Checks that the condition works if time update is higher than limit. """
        kwargs = dict(time_limit=2, time_update=10)
        start_time = time.time()
        controller = Controller(self.env, **kwargs)

        for each in controller:
            if (time.time() - start_time) >= (kwargs['time_limit'] + 0.2):
                self.fail("Not terminated within expected time.")

    def test_step_limit(self):
        kwargs = dict(step_limit=100, step_update=50)
        controller = Controller(self.env, **kwargs, short_summary=False)

        for env, (episodes, steps, time) in controller:
            env.step(np.zeros(self.instances))
            if steps > kwargs['step_limit']:
                self.fail("Not terminated within expected steps.")

    def test_step_limit_update(self):
        """ Checks that the condition works if step update is higher than limit. """
        kwargs = dict(step_limit=100, step_update=150)
        controller = Controller(self.env, **kwargs, short_summary=False)

        for env, (episodes, steps, time) in controller:
            env.step(np.zeros(self.instances))
            if steps > kwargs['step_limit']:
                self.fail("Not terminated within expected steps.")

    def test_episode_limit(self):
        kwargs = dict(episode_limit=5, episode_update=3)
        controller = Controller(self.env, **kwargs, short_summary=False)

        for env, (episodes, steps, time) in controller:
            env.step(np.zeros(self.instances))
            if episodes > kwargs['episode_limit']:
                self.fail("Not terminated within expected episodes.")

    def test_episode_limit_update(self):
        """ Checks that the condition works if step update is higher than limit. """
        kwargs = dict(episode_limit=5, episode_update=10)
        controller = Controller(self.env, **kwargs, short_summary=False)

        for env, (episodes, steps, time) in controller:
            _, _, dones, _ = env.step(np.zeros(self.instances))
            if episodes > kwargs['episode_limit']:
                self.fail("Not terminated within expected epsiodes.")

    def test_combinations(self):
        kwargs = dict(step_limit=500, episode_limit=10)
        controller = Controller(self.env, **kwargs, short_summary=False)

        for env, (episodes, steps, time) in controller:
            _, _, dones, _ = env.step(np.zeros(self.instances))

            if episodes > kwargs['episode_limit']:
                self.fail("Not terminated within expected epsiodes.")

            if steps > kwargs['step_limit']:
                self.fail("Not terminated within expected steps.")
            steps += 1
