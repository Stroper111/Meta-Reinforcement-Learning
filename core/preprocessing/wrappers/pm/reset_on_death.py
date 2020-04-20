import numpy as np

from core.games.pommerman.constants import Result
from core.preprocessing.wrappers.pm.base import BasePommermanWrapper


class PommermanResetOnDeath(BasePommermanWrapper):
    """ Early terminator when agent dies.  """

    def __init__(self, env, agent_idx):
        super().__init__(env)
        self.env = env
        self.agent_idx = agent_idx
        self._reward = 0

    def __getattr__(self, item):
        return getattr(self.env, item)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._reward += reward[self.agent_idx]

        if self.preprocess(obs):
            self.env._agents[self.agent_idx].episode_end(self._reward)
            info['result'] = Result.Loss
            done = True

        return obs, reward, done, info

    def reset(self) -> np.array:
        self._reward = 0
        return self.env.reset()

    def preprocess(self, obs: list) -> bool:
        obs_agent: dict = obs[self.agent_idx]
        return (self.agent_idx + 10) not in obs_agent['alive']
