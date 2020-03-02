from core.preprocessing.wrappers import BaseWrapper


class UnpackRGB(BaseWrapper):
    """ Converter for Procgen, so all images are already unpacked. """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.instances = env.num_envs

    def step(self, action):
        """ Return exactly the same environment, but unpack the 'rgb' dict.  """
        img, reward, done, info = self.env.step(action)
        return img['rgb'], reward, done, info

    def reset(self):
        img = self.env.reset()
        return img['rgb']
