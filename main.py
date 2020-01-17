import os
import sys
import argparse
import json

## Adding core directory, for terminal use
directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directory)

from core import BaseAgentMultiEnv, BaseAgentGym
from core.agents import HvassLabAgent


def main(setup=None):
    """ MultiEnv, only accepts procgen environments, can be multiple.  """
    setup = dict(bigfish=9) if setup is None else setup
    controller = BaseAgentMultiEnv(setup)
    return controller


def main_gym(setup=None):
    """ Gym controller, only accepts 1 gym environment.  """
    setup = {'CartPole-v1': 1} if setup is None else setup
    controller = BaseAgentGym(setup)
    return controller


def main_hvass_lab(setup=None):
    """ Manually created, currently only for 1 single gym environment.  """
    setup = {'Breakout-v0': 1} if setup is None else setup
    controller = HvassLabAgent(setup)
    return controller


if __name__ == '__main__':

    # Valid agents
    agents = dict(default=main, gym=main_gym, hvasslab=main_hvass_lab)

    # Description of this program.
    desc = "Reinforcement Learning (Q-learning) for Atari/Procgen Games using Keras."

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=desc)

    # Add arguments to the parser.
    parser.add_argument("--agents", required=False, default='default',
                        help=f"Agents that can be directly called from the command line, valid options are "
                             f"{' '.join(agents.keys())}")

    # Add arguments to the parser.
    parser.add_argument("--setup", required=True, default='Breakout-v0 1', nargs='+',
                        help="The setup environments as dictionary, where the key is he environment name and the value "
                             "are the number of instances of that environment, example: 'coinrun 1 bigfish 1', "
                             "this creates two environments, one coinrun and bighfish environment.")

    args = parser.parse_args()


    controller = agents[args.agents](args.setup)
    controller.run()
