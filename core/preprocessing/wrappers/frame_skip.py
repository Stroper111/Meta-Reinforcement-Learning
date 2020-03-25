import numpy as np
from core.preprocessing.wrappers import BaseWrapper


class FrameSkip(BaseWrapper):
    """
    Performs frames kipping on multi environments.

    All rewards are combined in these steps and if there is a game done
    the results of that game will be send.

    """

    def __init__(self, env, frame_skip: int):
        super().__init__(env)
        self.env = env
        self._frame_skip = frame_skip
        self._frame_counter = 0

        self.instances = self.env.instances

        self._actions = np.zeros((self.instances,), np.uint8)
        self._rewards = np.zeros((self.instances,))
        self._info = np.array([dict() for _ in range(self.instances)])
        self._done = np.zeros((self.instances,), dtype=np.bool)
        self._images = None  # Is set in _reset

    def step(self, actions):
        """ Return exactly the same environment, but unpack the 'rgb' dict.  """
        self._done.fill(0)
        self._rewards.fill(0)
        return self._skip(frame_skips=self._frame_skip, actions=actions)

    def reset(self):
        self._frame_counter = 0
        self._images = self.env.reset()
        self._skip(self._frame_skip, actions=self._actions)
        return self._images

    def _skip(self, frame_skips, actions):
        for _ in range(frame_skips):
            self._frame_counter += 1
            img, reward, done, info = self.env.step(actions)
            self._rewards += reward

            for idx in np.where(done)[0]:
                self._images[idx] = img[idx]
                self._done[idx] = True
                self._info[idx] = info[idx]

        return self._images, self._rewards, self._done, self._info
