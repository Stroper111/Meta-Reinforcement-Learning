import numpy as np
import PIL.Image

from core.preprocessing.wrappers import BaseWrapper


class RescalingGray(BaseWrapper):
    """
        Converts a gray image to a new dimension

        setup: dict
            The games as key and the instances as values , required for some wrappers.
        new_shape: [List, Tuple]
            The new x and y dimension of the image.

    """

    def __init__(self, setup, new_shape=(105, 80)):
        super().__init__(setup)

        # Size of each image in the state. Reversed order used by PIL.Image.
        self.new_shape = tuple(reversed(new_shape))

    def step(self, action):
        img, *args = self.env.step(action)
        img = self.process(img['rgb'])
        return (dict(rgb=img), *args)

    def reset(self):
        return dict(rgb=self.process(self.env.reset()['rgb']))

    def process(self, img: np.array):
        # Create PIL-object from numpy array.
        img = PIL.Image.fromarray(img)

        # Resize the image.
        img_resized = img.resize(size=self.new_shape,
                                 resample=PIL.Image.LINEAR)
        return img_resized
