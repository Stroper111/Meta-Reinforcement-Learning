import numpy as np
import time

from PIL import Image

from core.preprocessing.wrappers import BaseWrapper


class PongWrapper(BaseWrapper):
    """ Converter for MultiEnv generated images.  """
    _gray_values = np.array([0.2126, 0.7152, 0.00722])
    _cutoff = 0
    _crop_slice = slice(35, 195)
    _down_sampling_rate = 2

    def __init__(self, env):
        super().__init__(env)

        self.previous = None

    def step(self, action):
        img, *args = self.env.step(action)
        img = self.process(img)
        return (img, *args)

    def reset(self):
        return self.process(self.env.reset())

    def process(self, img: np.ndarray) -> np.ndarray:
        """ Process the image to be similar to the blog.  """
        # img = np.einsum("ijkl, l -> ijk", img, self._gray_values)
        img = self._crop_image(img, keep=self._crop_slice)
        img = self._down_sample(img, rate=self._down_sampling_rate)

        img[img == 144] = 0  # erase background (background type 1)
        img[img == 109] = 0  # erase background (background type 2)
        img[img != 0] = 1  # everything else (paddles, ball) just set to 1
        return np.array([each.astype(np.float).ravel() for each in img], dtype=np.float)

    def _crop_image(self, img: np.ndarray, keep: slice) -> np.ndarray:
        """ Slice part of the image.  """
        img = img[:, keep]
        return img

    def _down_sample(self, img: np.ndarray, rate: int) -> np.ndarray:
        """ Reduce size of the image"""
        img = img[:, ::rate, ::rate, 0]
        return img

    @staticmethod
    def show_image(image, wait_time):
        """ Helper to depict an image for a wait time number of seconds.  """
        image = np.hstack([*image])
        img = Image.fromarray(image)
        img.show()
        time.sleep(wait_time)
        img.close()

    def unravel(self, image):
        """ Reshapes the image back so it can be shown.  """
        img_x = self._crop_slice.stop - self._crop_slice.start
        img_y = 160  # Default Pong

        img_x = len([1 for _ in range(0, img_x, self._down_sampling_rate)])
        img_y = len([1 for _ in range(0, img_y, self._down_sampling_rate)])

        if image.size == img_x * img_y:
            return image.reshape(img_x, img_y, 1)

        if image.size == img_x * img_y * 3:
            return image.reshape(img_x, img_y, 3)
        raise ValueError("Image size is not Gray or RGB, please check dimensions?")


if __name__ == '__main__':
    # Testing code for the Preprocessing.
    from core import MultiEnv

    env = MultiEnv({"Pong-v0": 1})
    env = PongWrapper(env)

    # Get an example from the wrapped environment
    image = env.reset()

    for _ in range(200):
        image, *_ = env.step([0])

    PongWrapper.show_image(env.unravel(image), wait_time=0)
