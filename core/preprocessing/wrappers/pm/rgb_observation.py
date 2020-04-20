import os
import pathlib
import itertools
import numpy as np

from PIL import Image

from core.preprocessing.wrappers.pm.base import BasePommermanWrapper
from core.games.pommerman.constants import Item


class PommermanRGBObservation(BasePommermanWrapper):
    """ Add an extra observation to agent dictionary, `rgb_array`.  """

    def __init__(self, env, agent_idx: int):
        super().__init__(env)
        self.env = env
        self.agent_idx: int = agent_idx

        self._image_size: int = 48
        self._images_rgb: dict = self._init_images()
        self._last_obs = np.zeros((11 * self._image_size, 11 * self._image_size, 3), dtype=np.uint8)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs)
        return obs, reward, done, info

    def reset(self) -> np.array:
        obs = self.env.reset()
        obs[self.agent_idx] = self.preprocess(obs[self.agent_idx])
        return obs

    def preprocess(self, obs_agent: dict) -> dict:
        board = obs_agent['board']

        for x, y in itertools.product(range(11), range(11)):
            # Get base image
            image = self._images_rgb[board[y, x]]

            # Set timer on bomb images.
            if board[y, x] == Item.Bomb.value:
                bomb_life = obs_agent['bomb_life'][y, x]
                image = self._images_rgb[board[y, x]][bomb_life]

            y_coordinates = slice(y * self._image_size, (y + 1) * self._image_size)
            x_coordinates = slice(x * self._image_size, (x + 1) * self._image_size)
            self._last_obs[y_coordinates, x_coordinates, :] = image

        obs_agent['rgb_array'] = self._last_obs
        return obs_agent

    def _init_images(self):
        mapping_image = dict()

        # Store all values and names of items on the board.
        mapping_names = {item.value: item.name for item in Item}
        base = os.path.join(str(pathlib.Path(__file__).parents[3]), "games", "pommerman", "resources")

        # Set all base images
        for idx, item in mapping_names.items():
            mapping_image[idx] = self._load_image(os.path.join(base, f"{item}.png"))

        # Change bomb image depending on timer.
        mapping_image[Item.Bomb.value] = dict()

        for timer in range(1, 11):
            image_path = os.path.join(base, f"{Item.Bomb.name}-{timer}.png")
            mapping_image[Item.Bomb.value][timer] = self._load_image(image_path)

        return mapping_image

    def _load_image(self, path):
        """ Helper to load images dynamically.  """
        return np.array(Image.open(path).resize((self._image_size, self._image_size)))[:, :, :3]
