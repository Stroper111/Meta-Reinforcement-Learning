import numpy as np

from core.preprocessing.wrappers import BaseWrapper


class RGB2Gray(BaseWrapper):
    """ Converter for MultiEnv generated images.  """

    def __init__(self, env):
        super(RGB2Gray, self).__init__(env)
        self._gray_values = np.array([0.2126, 0.7152, 0.00722])

    def step(self, action):
        img, *args = self.env.step(action)
        img = self.process(img)
        return (img, *args)

    def reset(self):
        return self.process(self.env.reset())

    def process(self, img: np.array):
        gray = np.einsum("ijkl, l -> ijk", img, self._gray_values)
        return gray
