from core.preprocessing.abstract import AbstractPreProcessing
from core.preprocessing.wrappers import *

from core.tools import MultiEnv

# TODO Rebuild
class PreProcessingHvasslab(AbstractPreProcessing):
    """ Handles the Hvasslab preprocessing.

        env: MultiEnv, gym.make
            The environment to be wrapped
        gym: bool
            Indicator if the gym wrapper has to be applied or not.
        rescaling_dim: [Tuple, List]
            The new 2D dimensions of the image, only 2D image objects are allowed
            so RGB2Gray wrapper will automatically be applied.
        motion_tracer: bool
            Indicator if the motion tracer wrapper has to be applied.
        """

    def __init__(self, env, gym=False, rescaling_dim=None, motion_tracer=True,
                 statistics=True, history_size=30, save_dir=None):
        super().__init__(env)

        self.env = env
        self.instances = 1
        self.gym = gym
        self.rescaling_dim = rescaling_dim
        self.motion_tracer = True

        if gym:
            self.env = GymWrapper(self.env)
        else:
            self.instances = sum(self.env.setup.values())

        if rescaling_dim is not None:
            self.env = RGB2Gray(self.env)
            self.env = RescalingGray(self.env, rescaling_dim)

        if motion_tracer:
            self.env = MotionTracer(self.env)

        if statistics:
            assert save_dir is not None, "Need a saving directory for statistics."
            self.env = StatisticsUnique(self.env, history_size=history_size, save_dir=save_dir)

    def input_shape(self):
        shape = (210, 160, 3)
        if self.rescaling_dim is not None:
            shape = (*self.rescaling_dim, 1)

        if self.motion_tracer:
            shape = (*shape[:-1], 2)

        return shape

    def output_shape(self):
        if self.gym:
            return self.env.action_space.n
        return list(self.env.action_space.values())[0].n
