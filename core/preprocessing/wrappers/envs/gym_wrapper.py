import gym
import numpy as np

from typing import List
from collections import namedtuple

from .gym_base import BaseGymWrapper


class GymWrapper(BaseGymWrapper):
    """ Converter for gym to MultiEnv """
    _step = namedtuple("step", ("img", "reward", "done", "info"))

    def __init__(self, game, instances=1):
        super().__init__()
        self._env = self._create_envs(game, instances)
        self.game = game
        self.instances = instances

    def __getattr__(self, item):
        return [getattr(each, item) for each in self._env]

    def _create_envs(self, game, instances) -> List[gym.Env]:
        return [gym.make(game) for _ in range(instances)]

    def step(self, actions: np.array) -> (np.array, np.array, np.array, np.array):
        # IndexErrors, usually occurs whenever the action input is not correct.
        results = [self._step(*env.step(action)) for action, env in zip(actions, self._env)]
        results = self._step(*[np.stack(each) for each in zip(*results)])
        results = self._reset_done(results)

        # This handles 1D environments.
        images = results.img
        if images.ndim == 1:
            images = np.expand_dims(images, axis=1)

        return images, results.reward, results.done, results.info

    def reset(self) -> np.array:
        """ Expand dimensions of gym for stacking.  """
        images = np.stack([env.reset() for env in self._env])

        # This handles 1D environments.
        if images.ndim == 1:
            images = np.expand_dims(images, axis=1)

        return images

    def render(self):
        [env.render() for env in self._env]

    def close(self):
        [env.close() for env in self._env]
        self._env = None

    def _reset_done(self, results) -> _step:
        reset_idx = np.where(results.done)[0]
        for idx in reset_idx:
            img = np.array(self._env[idx].reset())
            results.img[idx] = img.copy()
        return results
