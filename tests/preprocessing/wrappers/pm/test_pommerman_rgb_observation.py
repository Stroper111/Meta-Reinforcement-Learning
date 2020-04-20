import numpy as np
import unittest
import pickle
import time
import os

from unittest.mock import Mock
from PIL import Image

from core.preprocessing.wrappers.pm import PommermanRGBObservation


class TestPommermanRGBObservation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = Mock()
        cls.agent_idx: int = 0

        cls.wrapper = PommermanRGBObservation(cls.env, agent_idx=cls.agent_idx)
        #
        current_directory = os.path.dirname(os.path.abspath(__file__))
        cls.save_path = os.path.join(current_directory, "..", "data", "_pommerman_rgb.pkl")

        with open(cls.save_path, "rb") as file:
            cls.obs_saved = pickle.load(file)

    def test_images(self):
        board = np.zeros((11, 11))
        board[0] = np.arange(1, 2)
        board[1][1:] = np.arange(4, 14)
        board[2] = np.ones((11,)) * 3  # Bombs

        # Increasing life from 0 to 10, 10.
        bomb_life = np.array([[*np.arange(1, 11), 10] for _ in range(0, 11)])

        obs = dict(board=board, bomb_life=bomb_life)
        obs_processed = self.wrapper.preprocess(obs)[self.agent_idx]
        self.assertEqual(True, np.array_equal(obs_processed['rgb_array'], self.obs_saved))

    @staticmethod
    def _show_image(image, wait_time):
        """ Helper to depict an image for a wait time number of seconds.  """
        img = Image.fromarray(image)
        img.show()
        time.sleep(wait_time)
        img.close()

    @staticmethod
    def _save_array(path, array):
        """ Helper function to store numpy array for test.  """
        with open(path, " wb") as file:
            pickle.dump(array, file)
