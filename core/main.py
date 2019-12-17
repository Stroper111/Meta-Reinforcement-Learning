import os
import sys

directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)


if __name__ == '__main__':
    from core import BaseAgent

    controller = BaseAgent()
    controller.run()
