import os
import sys

directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(directory)

from core import BaseAgent

if __name__ == '__main__':
    controller = BaseAgent()
    controller.run()
