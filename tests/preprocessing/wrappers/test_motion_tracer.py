import unittest
import pickle
import numpy as np
import time
import os

from PIL import Image

from core.preprocessing.wrappers.motion_tracer import MotionTracer


class TestMotionTracer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup = dict(coinrun=12)
        cls.wrapper = MotionTracer(cls.setup)

        current_directory = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current_directory, "_original_images.pkl"), "rb") as file:
            cls.images = pickle.load(file)['rgb']

        with open(os.path.join(current_directory, "_original_images_motion_traced.pkl"), "rb") as file:
            cls.image_processed = pickle.load(file)

    def test_process_single_input(self):
        motion_traced = []
        motion_traced_display = []
        for img in self._rgb2_gray(self.images):
            motion_trace = self.wrapper.process(img)
            motion_traced.append(motion_trace)

            # uncomment for displaying on the screen, this is the easiest way
            motion_traced_display.append(np.hstack([*motion_trace.transpose(2, 0, 1)]))
        self._show_image(np.array(motion_traced_display).reshape(64 * 12, 128))

        motion_traced = np.array(motion_traced)
        self.assertEqual((12, 64, 64, 2), motion_traced.shape)
        self.assertEqual(True, np.array_equal(self.image_processed, motion_traced),
                         "Processed image is not the same as stored")

    def test_process_multiple_input(self):
        images = self._rgb2_gray(self.images)
        motion_trace = self.wrapper.process(images)

        motion_traced = np.array([np.hstack([np.transpose(img, axes=(1, 2, 0))]) for img in motion_trace])
        self._show_image(motion_traced.reshape(64*12, 64*2))

        self.assertEqual((12, 64, 64, 2), motion_trace.shape)
        self.assertEqual(True, np.array_equal(self.image_processed, motion_trace),
                         "Processed image is not the same as stored")

    def _show_saved_image(self):
        """ Not a test method, but created for when a user wants to see input/output.  """
        image = np.reshape(self.images, newshape=(64 * 3, 64 * 4, 3))
        self._show_image(image, wait_time=1)

    def _save_image(self, image):
        """ Helper function to store manually checked image after testing, please use wisely.  """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(current_directory, "_original_images_motion_traced.pkl"), "wb") as file:
            pickle.dump(image, file)

    @staticmethod
    def _show_image(image, wait_time=1):
        """ Helper to depict an image for a wait time number of seconds.  """
        img = Image.fromarray(image).convert('LA')
        img.show()
        time.sleep(wait_time)
        img.close()

    def _rgb2_gray(self, img):
        gray = np.einsum("ijkl, l -> ijk", img, np.array([0.2126, 0.7152, 0.00722]))
        return gray


if __name__ == '__main__':
    unittest.main()
