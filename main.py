import os
import sys
import argparse

## Adding core directory, for terminal use
directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directory)

from core.agents import CartPole

if __name__ == '__main__':
    setup = {"CartPole-v0": 1}
    controller = CartPole(setup)
    controller.run()
