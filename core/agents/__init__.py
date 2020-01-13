try:
    import gym
except ModuleNotFoundError:
    print("Module gym not found, please use 'pip install gym'.")

try:
    import procgen
except ModuleNotFoundError:
    print("Module procgen not found, pleasue use 'pip install procgen'.")

from .abstract import AbstractAgent
from .base import BaseAgent
from .base_multi_env import BaseAgentMultiEnv
from .base_gym import BaseAgentGym
from .hvass_lab import HvassLab

__all__ = ['AbstractAgent', 'BaseAgent', 'BaseAgentMultiEnv', 'BaseAgentGym',
           'HvassLab']
