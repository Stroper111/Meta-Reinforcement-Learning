import numpy as np

from core.preprocessing.wrappers import BaseWrapper


class GymWrapper(BaseWrapper):
    """ Converter for gym to MultiEnv """
    def __init__(self, env):
        super().__init__(env)
        self.on_reset = False
        self.setup = {env.spec.id: 1}
        self.instances = 1
        self.venv = [env]

    def step(self, action):
        """ Perform a step in a gym env and wraps everything like a MultiEnv.  """
        if self.on_reset:
            return self._on_reset()

        img, reward, done, info = self.env.step(action)
        self.on_reset = done
        return self._image(img), np.array(reward), np.array(reward), np.array(reward)

    def reset(self):
        """ Wraps the reset.  """
        return dict(img=self._image(self.env.reset()))

    def _on_reset(self):
        """ Procgen envs don't reset, they play on.  This creates the same effect.  """
        self.on_reset = False
        return self.reset(), np.zeros(1), np.array(False), np.zeros(1)

    def _image(self, image):
        """ Converts image to look like MultiEnv.  """
        return dict(img=np.expand_dims(image, axis=0))
