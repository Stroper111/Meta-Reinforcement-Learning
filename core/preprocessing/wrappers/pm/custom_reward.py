import numpy as np

from core.games.pommerman.constants import Item
from core.preprocessing.wrappers.pm.base import BasePommermanWrapper


class PommermanReward(BasePommermanWrapper):
    """
        Create a custom reward function for the agent.

        :param env: The environment from which we will get the observation
        :param agent_idx: The position of the reward we have to change
        :param kills: The points obtained when killing a unit
        :param boxes: The points obtained when destroying a box
        :param powerups: The points obtained when picking up a powerup
        :param bombs: The points gained for placing a bomb
        :param alive: The points gained for staying alive another step
    """

    def __init__(self, env, agent_idx=0, kills: float = 0., boxes: float = 0.,
                 powerups: float = 0., bombs: float = 0., alive: float = 0.):
        super().__init__(env)
        self.env = env
        self.agent_idx = agent_idx

        self._rewards = dict(kills=kills, boxes=boxes, powerups=powerups, bombs=bombs, alive=alive)
        self.clean_rewards()

        # Forward declarations
        self._bombs = None
        self._bombs_placed = None
        self._blast_strength = None
        self._boxes = None
        self._can_kick = None
        self._enemies = None
        self._reward_counts = None

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def reward_counts(self) -> str:
        return ',\t'.join(['%s %3d' % (key, value) for key, value in self._reward_counts.items()])

    def clean_rewards(self):
        """ Remove all zero rewards, since these are not interesting.  """
        remove = []
        for key, value in self._rewards.items():
            if not value:
                remove.append(key)

        for key in remove:
            self._rewards.pop(key)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward[self.agent_idx] = self.preprocess(obs[self._agent_idx], reward[self.agent_idx])
        return obs, reward, done, info

    def reset(self) -> np.array:
        obs = self.env.reset()
        self._init_values(obs[self.agent_idx])
        return obs

    def preprocess(self, obs: list, reward) -> int:
        agent_obs = obs[self.agent_idx]

        for key, value in self._rewards.items():
            reward += getattr(self, key)(agent_obs) * value

        return reward

    def _init_values(self, obs):
        """ Set all initial values.  """
        self._bombs = obs['ammo']
        self._bombs_placed = 0
        self._blast_strength = obs['blast_strength']
        self._boxes = len(np.where(obs['board'] == Item.Wood.value)[0])
        self._can_kick = obs['can_kick']
        self._enemies = len(obs['enemies'])
        self._reward_counts = dict(kills=0, boxes=0, powerups=0, bombs=0, alive=0)

    def kills(self, obs) -> float:
        """ Calculate the number of destroyed enemies.  """
        if len(obs['enemies']) != self._enemies:
            kills = (self._enemies - len(obs['enemies']))
            self._reward_counts['kills'] += kills
            return 1 * kills
        return 0.

    def boxes(self, obs) -> float:
        """ Calculate the number of destroyed boxes.  """
        boxes = len(np.where(obs['board'] == Item.Wood.value)[0])
        if boxes != self._boxes:
            destroyed = self._boxes - boxes
            self._boxes = boxes
            self._reward_counts['boxes'] += destroyed
            return 1 * (destroyed)
        return 0.

    def powerups(self, obs) -> float:
        """ Calculate the reward if a powerup is picked up.  """
        reward = 0.

        new_ammo = obs['ammo']
        new_can_kick = obs['can_kick']
        new_strength = obs['blast_strength']

        # Picked up a can kick powerup.
        if new_can_kick != self._can_kick:
            self._can_kick = new_can_kick
            reward += 1.

        # Picked up an extra bomb
        if new_ammo > self._bombs + self._bombs_placed:
            self._bombs = new_ammo
            reward += 1.

        # Picked up increased blast strength
        if new_strength > self._blast_strength:
            self._blast_strength = new_strength
            reward += 1.

        if reward:
            self._reward_counts['powerups'] += reward

        return reward

    def bombs(self, obs) -> float:
        """ Return reward for placing an extra bomb.  """
        bombs_placed = len(np.where(obs['bomb_life'] > 0)[0])

        if bombs_placed > self._bombs_placed:
            self._bombs_placed = bombs_placed
            self._reward_counts['bombs'] += 1
            return 1.

        self._bombs_placed = bombs_placed
        return 0.

    def alive(self, obs) -> float:
        """ Return a reward for being alive.  """
        self._reward_counts['alive'] += 1
        return 1
