import unittest
import pickle
import numpy as np
import time

from PIL import Image

from core.preprocessing.wrappers import RGB2Gray


class TestRGB2Gray(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup = dict(coinrun=12)
        cls.wrapper = RGB2Gray(cls.setup)

        with open("test_images.pkl", "rb") as file:
            cls.images = pickle.load(file)['rgb']

        with open("test_images_black.pkl", "rb") as file:
            cls.image_processed = pickle.load(file)

    def test_process(self):
        black = self.wrapper.process(self.images)
        image = np.reshape(black, newshape=(64 * 3, 64 * 4))
        self.assertEqual(True, np.array_equal(self.image_processed, image), "Processed image is not the same as stored")

    def _show_saved_image(self):
        """ Not a test method, but created for when a user wants to see input/output.  """
        image = np.reshape(self.images, newshape=(64 * 3, 64 * 4, 3))
        self._show_image(image, wait_time=3)

    @staticmethod
    def _show_image(image, wait_time):
        """ Helper to depict an image for a wait time number of seconds.  """
        img = Image.fromarray(image)
        img.show()
        time.sleep(wait_time)
        img.close()


if __name__ == '__main__':
    unittest.main()
