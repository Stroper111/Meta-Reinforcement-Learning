import numpy as np

from core.preprocessing.wrappers import BaseWrapper


class UnpackVec(BaseWrapper):
    """
    Converter from vector to a single gym instance.
    This unpacks all np.arrays, this means that for a model
    you will manually have to increase the dimensions again and
    that this wrapper can only be used on single instances.
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.instances = env.instances

    def step(self, action):
        """ Return exactly the same environment, but unpack the vector.  """
        img, reward, done, info = [each[0] for each in self.env.step(action)]
        return img, reward, done, info

    def reset(self):
        img = self.env.reset()
        assert next(iter(img.shape)) == 1, "There is more than one environment."
        return np.squeeze(img, axis=(0,))
