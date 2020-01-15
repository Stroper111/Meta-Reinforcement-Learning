import numpy as np

from core.preprocessing.wrappers import BaseWrapper


class GymWrapper(BaseWrapper):
    """ Converter for gym to MultiEnv """

    def __init__(self, env):
        super().__init__(env)
        self._on_reset = False
        self.setup = {env.spec.id: 1}
        self.instances = 1
        self.venv = [env]
        self._reset_values = self._setup_reset_values(env)

    def step(self, action):
        """ Perform a step in a gym env and wraps everything like a MultiEnv.  """
        if self._on_reset:
            return self._on_reset()

        img, reward, done, info = self.env.step(action[0])
        self._on_reset = done
        return self._image(img), np.array([reward]), np.array([done]), [info]

    def reset(self):
        """ Wraps the reset.  """
        return self._image(self.env.reset())

    def _on_reset(self):
        """ Procgen envs don't reset, they play on.  This creates the same effect.  """
        self._on_reset = False
        return (self.reset(), *self._reset_values,)

    def _setup_reset_values(self, env):
        env.reset()
        _, reward, done, info = env.step(0)
        return np.array([0]), np.array([0]), [info]

    def _image(self, image):
        """ Converts image to look like MultiEnv.  """
        return dict(rgb=np.expand_dims(image, axis=0))
