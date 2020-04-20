from core.preprocessing.wrappers.envs.gym_base import BaseGymWrapper
from core.games.pommerman.envs.v0 import Pomme


class BasePommermanWrapper(BaseGymWrapper):
    env: Pomme

    def __init__(self, env, *args, **kwargs):
        super().__init__()

        self.env = env

    def act(self, obs):
        return self.env.act(obs)

    def preprocess(self, *args, **kwargs):
        raise NotImplementedError
