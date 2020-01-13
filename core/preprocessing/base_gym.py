from core.preprocessing import AbstractPreProcessing
from core.preprocessing.wrappers import RGB2Gray, FrameStack, StatisticsUnique, GymWrapper


class BasePreProcessingGym(AbstractPreProcessing):
    def __init__(self, env,
                 rgb2gray=False,
                 frame_stack=0,
                 statistics=True, history_size=30, save_dir=None):

        self.env = env
        self.env = GymWrapper(self.env)

        self.instances = self.env.instances
        self.rgb2gray = rgb2gray
        self.frame_stack = frame_stack

        if rgb2gray:
            self.env = RGB2Gray(self.env)

        if frame_stack and frame_stack > 0:
            assert isinstance(frame_stack, int), "The number has to be an integer"
            self.env = FrameStack(self.env, stack=frame_stack)

        if statistics:
            assert save_dir is not None, "Need a saving directory for statistics."
            self.env = StatisticsUnique(self.env, history_size=history_size, save_dir=save_dir)

    def input_shape(self):
        """ Calculate input shape.  """
        shape = self.env.observation_space.shape

        if self.rgb2gray:
            shape = shape[:-1]

        if self.frame_stack and self.frame_stack > 0:
            shape = (*shape, self.frame_stack)
        return shape

    def output_shape(self):
        return self.env.action_space.n
