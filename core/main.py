import os
import sys
sys.path.append(os.getcwd())


from core import BaseAgent

if __name__ == '__main__':
    controller = BaseAgent()
    controller.run()
