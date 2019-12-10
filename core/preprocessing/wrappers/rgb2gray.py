import numpy as np

from core.preprocessing.wrappers import BaseWrapper


class RGB2Gray(BaseWrapper):
    """ Converter for MultiEnv generated images.  """

    def __init__(self, setup):
        super(RGB2Gray, self).__init__(setup)
        self.gray = np.array([0.2126, 0.7152, 0.00722])

    def step(self, action):
        img, *args = self.env.step(action)
        img = self.process(img['rgb'])
        return (dict(rgb=img), *args)

    def reset(self):
        return dict(rgb=self.process(self.env.reset()['rgb']))

    def process(self, img: np.array):
        gray = np.einsum("ijkl, l -> ijk", img, self.gray)
        return gray
