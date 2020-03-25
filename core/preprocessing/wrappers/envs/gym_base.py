import gym
import procgen
import numpy as np

from typing import Union
from abc import abstractmethod


class BaseGymWrapper:
    """
        Wrapper for processing environment images from the MultiEnv
    """
    env: gym.Env

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

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

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def process(self, img: np.array):
        raise NotImplementedError
