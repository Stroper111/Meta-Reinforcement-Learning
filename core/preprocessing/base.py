from core.preprocessing import AbstractPreProcessing


class BasePreProcessingGym(AbstractPreProcessing):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def input_shape(self):
        """ Calculate input shape.  """
        return tuple

    def output_shape(self):
        return int
