try:
    import gym
except ModuleNotFoundError:
    print("Module gym not found, please use 'pip install gym'.")

try:
    import procgen
except ModuleNotFoundError:
    print("Module procgen not found, please use 'pip install procgen'.")

import os, sys

## Adding core directory, for terminal use
directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)
