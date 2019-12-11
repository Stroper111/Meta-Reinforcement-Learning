from core.tools import MultiEnv
from core.preprocessing.wrappers import RGB2Gray, FrameStack, StatisticsUnique


class BasePreProcessing:
    def __init__(self, env: MultiEnv,
                 rgb2gray=True,
                 frame_stack=4,
                 statistics=True, history_size=30, save_dir=None):

        self.env = env

        if rgb2gray:
            self.env = RGB2Gray(self.env)

        if frame_stack and frame_stack > 0:
            assert isinstance(frame_stack, int), "The number has to be an integer"
            self.env = FrameStack(self.env, stack=frame_stack)

        if statistics:
            assert save_dir is not None, "Need a saving directory for statistics."
            self.env = StatisticsUnique(self.env, history_size=history_size, save_dir=save_dir)
