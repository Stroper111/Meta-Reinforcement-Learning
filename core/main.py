import os
import sys

## Adding core directory, for terminal use
directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

from core import BaseAgent, BaseAgentGym


def main():
    controller = BaseAgent()
    controller.run()


def main_gym():
    setup = {'CartPole-v0': 1}
    controller = BaseAgentGym(setup)
    controller.run()


if __name__ == '__main__':
    main_gym()
