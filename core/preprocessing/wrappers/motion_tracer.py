import numpy as np

from core.preprocessing.wrappers import BaseWrapper


class MotionTracer(BaseWrapper):
    """
        Converts a gray image to a new dimension

        setup: dict
            The games as key and the instances as values , required for some wrappers.
        decay: float
            Parameter for how long the tail should be on the motion-trace.
            This is a float between 0.0 and 1.0 where higher values means
            the trace / tail is longer.
    """

    def __init__(self, env, decay=0.75):
        super().__init__(env)

        # Size of each image in the state. Reversed order used by PIL.Image.
        self._setup = False
        self._decay = decay

        self._last_input = np.ndarray
        self._last_output = np.ndarray
        self._dimensions = int

    def step(self, action):
        img, *args = self.env.step(action)
        img = self.process(img)
        return (img, *args)

    def reset(self):
        self._setup = False
        return self.process(self.env.reset())

    def process(self, images: np.array):
        if not self._setup:
            self._run_setup(images)

        # Calculate difference
        img_dif = images - self._last_input

        self._last_input[:] = images[:]

        # Execute a threshold
        img_motion = np.where(np.abs(img_dif) > 20, 255., 0)

        # Calculate motion trace
        output = img_motion + self._decay * self._last_output

        # Clip input to valid values
        output = np.clip(output, 0.0, 255.0)

        # Store output value
        self._last_output = output

        return self._image()

    def _run_setup(self, images):
        """ For new games setup the correct motion trace.  """
        self._last_input = images.astype(np.float)
        self._last_output = np.zeros_like(images)
        self._dimensions = np.arange(0, len(self._last_input.shape) + 1)
        self._setup = True

    def _image(self):
        """ Return neural network input.  """

        # Stack the last input and output images.
        state = np.stack([self._last_input, self._last_output])
        state = np.transpose(state, axes=(*self._dimensions[1:], 0))

        # Convert to 8-bit integer.
        # This is done to save space in the replay-memory.
        state = state.astype(np.uint8)

        return state
