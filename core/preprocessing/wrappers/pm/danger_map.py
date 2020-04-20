import numpy as np

from typing import List

from .base import BasePommermanWrapper
from core.games.pommerman.envs.v0 import Pomme


class PommermanDangerMap(BasePommermanWrapper):
    """ Add an extra observation to agent dictionary, `danger_map` """

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
        obs_agent['danger_map'] = self.create_danger_map(obs_agent)
        return obs_agent

    @staticmethod
    def create_danger_map(obs: dict) -> np.ndarray:
        # Set our initial danger map
        danger_map = obs['flame_life']

        # Find all bomb locations, bomb timers and strength
        bombs = np.where(obs['bomb_life'] > 0)
        bombs_timers = map(int, obs['bomb_life'][bombs])
        bomb_strength = map(int, obs['bomb_blast_strength'][bombs])

        # Now we are going to set the danger information
        combined_info = zip(*bombs, bombs_timers, bomb_strength)
        for row, col, timer, strength in sorted(combined_info, key=lambda x: x[1], reverse=True):

            # Reduce strength by one, since we are creating a `+` around the center.
            strength -= 1

            # Calculate the upper and lower ranges of the bombs (this is the + sign).
            row_low, row_high = max(row - strength, 0), min(row + strength, 10)
            col_low, col_high = max(col - strength, 0), min(col + strength, 10)

            # Set the information on the danger map, first row and then column.
            # Note that we have presorted on bomb timers, so lowest counts are maintained.
            for row_danger in range(row_low, row_high + 1):
                danger_map[row_danger, col] = timer

            for col_danger in range(col_low, col_high + 1):
                danger_map[row, col_danger] = timer

        return danger_map
