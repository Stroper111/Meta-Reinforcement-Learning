import numpy as np

from typing import List

from .base import BasePommermanWrapper
from core.games.pommerman.envs.v0 import Pomme


class PommermanUnpack(BasePommermanWrapper):
    """ Converter for Pommerman observations """

    def __init__(self, env, agent_idx):
        super().__init__(env)
        self.env: Pomme = env
        self.agent_idx = agent_idx

    def __getattr__(self, item):
        return getattr(self.env, item)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs[self.agent_idx] = self.preprocess(obs[self.agent_idx])
        return obs, reward, done, info

    def reset(self) -> np.array:
        obs = self.env.reset()
        obs[self.agent_idx] = self.preprocess(obs[self.agent_idx])
        return obs

    def preprocess(self, obs_agent: dict) -> dict:
        results = []

        # Get all boards
        for key in ['board', 'bomb_blast_strength', 'bomb_life', 'bomb_moving_direction', 'flame_life']:
            results.append(obs_agent[key].flatten())

        # Get all fixed size meta data
        for key in ['position', 'blast_strength', 'can_kick', 'ammo']:
            results.append(obs_agent[key])

        # Get all variable length meta data (make them fixed length)
        for key in ['alive', 'enemies']:
            results.append((obs_agent[key] + [0] * 4)[:4])

        obs_agent['flatten'] = np.hstack(results)
        return obs_agent

    def _print_obs(self, obs):
        """ Helper function to print out a single observation."""
        for key, items in obs.items():
            if hasattr(items, 'shape'):
                print('\t%-25s, shape: %s' % (key, str(items.shape)))
            else:
                print('\t%-25s, item: %s' % (key, str(items)))
