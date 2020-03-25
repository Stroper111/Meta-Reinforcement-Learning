from core.preprocessing import AbstractPreProcessing

from core.tools import MultiEnv
from core.preprocessing.wrappers import RGB2Gray, FrameStack


class BasePreProcessingMultiEnv(AbstractPreProcessing):
    """
        Base processor for the MultiEnv, which is meant to play on the procgen module.

        Otherwise you might to correct the input_shape method.
    """

    def __init__(self, env: MultiEnv,
                 rgb2gray=True,
                 frame_stack=4,
                 statistics=True, history_size=30, save_dir=None):
        super().__init__(env)

        self.env = env
        self.instances = env.instances
        self.rgb2gray = rgb2gray
        self.frame_stack = frame_stack

        if rgb2gray:
            self.env = RGB2Gray(self.env)

        if frame_stack and frame_stack > 0:
            assert isinstance(frame_stack, int), "The number has to be an integer"
            self.env = FrameStack(self.env, stack=frame_stack)

    def input_shape(self):
        """ Calculate input shape.  """
        shape = (64, 64, 3)

        if self.rgb2gray:
            shape = shape[:-1]

        if self.frame_stack and self.frame_stack > 0:
            shape = (*shape, self.frame_stack)
        return shape

    def output_shape(self):
        return list(self.env.action_space.values())[0].n
