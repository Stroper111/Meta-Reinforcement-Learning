import numpy as np

from core.memory.sampling import BaseSampling


class BaseSamplingGym(BaseSampling):

    def reformat_states(self, states):
        """ Gym env doesn't need any reformatting.  """
        return np.array(states)
