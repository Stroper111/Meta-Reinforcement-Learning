import numpy as np

from collections import deque

from core.preprocessing.wrappers import BaseWrapper


class FrameStack(BaseWrapper):
    """ Converter for MultiEnv generated images.  """

    def __init__(self, env, stack=4):
        super().__init__(env)
        self._stack = stack
        self._frames = [deque([], maxlen=self._stack) for _ in range(self.env.instances)]

    def reset(self):
        img = self.env.reset()['rgb']
        self._create_stacks(img)
        return dict(rgb=self._get_images())

    def step(self, actions):
        images, reward, done, info = self.env.step(actions)
        images = np.expand_dims(images['rgb'], axis=1)
        for idx, image in enumerate(images):
            self._frames[idx].append(image)
        return dict(rgb=self._get_images()), reward, done, info

    def _create_stacks(self, images):
        images = np.expand_dims(images, axis=1)
        for idx, image in enumerate(images):
            for _ in range(self._stack):
                self._frames[idx].append(image)

    def _get_images(self):
        return [LazyFrames(list(frames)) for frames in self._frames]


class LazyFrames(object):
    def __init__(self, frames):
        """
            This object ensures that common frames between the observations are only stored once.
            It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
            buffers.

            This object should only be converted to numpy array before being passed to the model.
            You'd not believe how complex the previous solution was.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
