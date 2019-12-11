import unittest

from core import MultiEnv
from core.tools import Scheduler


class TestMultiEnv(unittest.TestCase):
    def setUp(self):
        self.setup = dict(coinrun=2, bigfish=2, chaser=2)
        self.instances = sum(self.setup.values())

        self.env = MultiEnv(self.setup)
        self.env_action_space = self.env.action_space['coinrun'].n

    def tearDown(self) -> None:
        self.env.close()