import unittest
import pickle
import numpy as np
import time
import os

from PIL import Image
from unittest.mock import Mock
from core.preprocessing.wrappers.motion_tracer import MotionTracer


class TestMotionTracer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        current_directory = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current_directory, "_original_images.pkl"), "rb") as file:
            cls.images = pickle.load(file)['rgb']

        with open(os.path.join(current_directory, "_original_images_motion_traced.pkl"), "rb") as file:
            cls.image_processed = pickle.load(file)

    def setUp(self) -> None:
        self.env = Mock(0)
        self.wrapper = MotionTracer(self.env)

    def test_process_single_input(self):
        motion_traced = []
        for img in self._rgb2_gray(self.images):
            motion_trace = self.wrapper.process(img)
            motion_traced.append(motion_trace)
            self.wrapper._run_setup(img)
        motion_traced = np.array(motion_traced)

        self.assertEqual((12, 64, 64, 2), motion_traced.shape)
        self.assertEqual(True, np.array_equal(self.image_processed, motion_traced),
                         "Processed image is not the same as stored")

    def test_process_multiple_input(self):
        images = self._rgb2_gray(self.images)
        motion_trace = np.array
        for shift in range(2):
            motion_trace = self.wrapper.process(np.roll(images[:], axis=0, shift=shift))

        # Set first image to dark
        originals = []
        originals_roll = []
        originals_processed = []

        # Setup the comparing test, this is compansiting compensate for the np.roll we did.
        for idx, (result, processed) in enumerate(zip(motion_trace, self.image_processed)):
            result = np.transpose(result, axes=(2, 0, 1))
            processed = np.transpose(processed, axes=(2, 0, 1))

            # The first image result should be black to be compatible with the single input test.
            if idx == 0:
                result[1] = np.zeros_like(result[1])

            originals.append((processed[0]))
            originals_roll.append(result[0])
            originals_processed.append((result[1], processed[1]))

        #  Correct for the roll we did
        originals_roll = np.roll(np.array(originals_roll), axis=0, shift=-1)
        originals = [(result, truth) for (result, truth) in zip(originals, originals_roll)]

        # Perform the actual check
        for (truth, processed) in originals + originals_processed:
            self.assertEqual(True, np.array_equal(truth, processed),
                             "Processed image is not the same as stored, you can use 'self._show_image()' for debug.")
        self.assertEqual((12, 64, 64, 2), motion_trace.shape)

    def _show_result_images(self, images):
        """ Helper function to show results next to each other.  """
        motion_traced_display = []
        for img in images:
            # uncomment for displaying on the screen, this is the easiest way
            motion_traced_display.append(np.hstack([*img.transpose(2, 0, 1)]))
        self._show_image(np.array(motion_traced_display).reshape(64 * len(images), 128))

    def _show_saved_image(self):
        """ Not a test method, but created for when a user wants to see input/output.  """
        image = np.reshape(self.images, newshape=(64 * 3, 64 * 4, 3))
        self._show_image(image, wait_time=1)

    @staticmethod
    def _save_image(image):
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
