import os
import sys

## Adding core directory, for terminal use
directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

from core import BaseAgentMultiEnv, BaseAgentGym
from core.agents import HvassLabAgent


def main(setup=None):
    """ MultiEnv, only accepts procgen environments, can be multiple.  """
    setup = dict(bigfish=9) if setup is None else setup
    controller = BaseAgentMultiEnv(setup)
    return controller


def main_gym():
    """ Gym controller, only accepts 1 gym environment.  """
    setup = {'CartPole-v1': 1}
    controller = BaseAgentGym(setup)
    return controller


def main_hvass_lab():
    """ Manually created, currently only for 1 single gym environment.  """
    setup = {'MsPacman-v0': 1}
    controller = HvassLabAgent(setup)
    return controller


if __name__ == '__main__':
    controller = main_hvass_lab()
    controller.run()
