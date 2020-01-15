import gym
import numpy as np

from typing import Union
from core.tools import MultiEnv


class BaseWrapper:
    """
        Wrapper for processing environment images from the MultiEnv
    """

    def __init__(self, env: Union[MultiEnv, gym.Env, 'BaseWrapper']):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def spec(self):
        return self.env.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def process(self, img: np.array):
        raise NotImplementedError
