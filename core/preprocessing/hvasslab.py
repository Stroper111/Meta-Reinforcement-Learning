
from core.preprocessing.abstract import AbstractPreProcessing
from core.preprocessing.wrappers import GymWrapper, RGB2Gray, RescalingGray, MotionTracer

from core.tools import MultiEnv


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

    def __init__(self, env, gym=False, rescaling_dim=None, motion_tracer=True):
        super().__init__(env)
        self.env = env
        self.instances = 1
        self.gym = gym
        self.rescaling_dim = rescaling_dim
        self.motion_tracer = True

        if gym:
            self.env = GymWrapper(env)
        else:
            self.instances = sum(self.env.setup.values())

        if rescaling_dim is not None:
            self.env = RGB2Gray(env)
            self.env = RescalingGray(env, rescaling_dim)

        if motion_tracer:
            self.env = MotionTracer(env)

    def input_shape(self):
        shape = (self.instances, 210, 160, 3)
        if self.rescaling_dim is not None:
            instances, x, y, z = shape
            shape = (instances, *self.rescaling_dim, 1)

        if self.motion_tracer:
            shape = (*shape[:-1], 2)

        return shape

    def output_shape(self):
        if self.gym:
            return self.env.action_space.n
        return list(self.env.action_space.values())[0].n
