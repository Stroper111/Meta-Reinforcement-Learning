import unittest
import pickle
import numpy as np
import time
import os

from PIL import Image

from core.preprocessing.wrappers.rescaling_gray import RescalingGray


class TestRescaling(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup = dict(coinrun=12)
        cls.new_shape = (200, 40)
        cls.wrapper = RescalingGray(cls.setup, new_shape=cls.new_shape)

        current_directory = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current_directory, "_original_images.pkl"), "rb") as file:
            cls.images = pickle.load(file)['rgb']

        with open(os.path.join(current_directory, "_original_images_rescale.pkl"), "rb") as file:
            cls.image_processed = pickle.load(file)

    def test_process(self):
        gray = self._rgb2_gray(self.images)
        rescaled = self.wrapper.process(gray)
        x, y = self.new_shape
        image = np.reshape(rescaled, newshape=(x * 3, y * 4))
        self.assertEqual((x * 3, y * 4), image.shape, "Processed image shape is not correct")
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

    def _rgb2_gray(self, img):
        gray = np.einsum("ijkl, l -> ijk", img, np.array([0.2126, 0.7152, 0.00722]))
        return gray


if __name__ == '__main__':
    unittest.main()
