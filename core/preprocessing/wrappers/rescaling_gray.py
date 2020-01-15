import numpy as np
import PIL.Image

from core.preprocessing.wrappers import BaseWrapper


class RescalingGray(BaseWrapper):
    """
        Converts a Grayscale image to a new dimension

        setup: dict
            The games as key and the instances as values , required for some wrappers.
        new_shape: [List, Tuple]
            The new x and y dimension of the image.

    """

    def __init__(self, setup, new_shape):
        super().__init__(setup)
        assert len(new_shape) == 2, "Only 2D images are supported at this time. (use np.squeeze)"

        # Size of each image in the state. Reversed order used by PIL.Image.
        x, y, *z = new_shape
        self.new_shape = (y, x, *z)
        self.new_image = np.zeros((sum(setup.values()), *new_shape))

    def step(self, action):
        img, *args = self.env.step(action)
        img = self.process(img['rgb'])
        return (dict(rgb=img), *args)

    def reset(self):
        return dict(rgb=self.process(self.env.reset()['rgb']))

    def process(self, images: np.array):
        for idx, img in enumerate(images):
            # Create PIL-object from numpy array.
            img = PIL.Image.fromarray(img)

            # Resize the image.
            img_resized = img.resize(size=self.new_shape,
                                     resample=PIL.Image.LINEAR)
            self.new_image[idx] = img_resized
        return self.new_image

