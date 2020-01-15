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

    def __init__(self, setup, decay=0.75):
        super().__init__(setup)

        # Size of each image in the state. Reversed order used by PIL.Image.
        self.setup = False
        self.decay = decay

        self.last_input = None
        self.last_output = None

    def step(self, action):
        img, *args = self.env.step(action)
        img = self.process(img['rgb'])
        return (dict(rgb=img), *args)

    def reset(self):
        self.setup = False
        return dict(rgb=self.process(self.env.reset()['rgb']))

    def process(self, img: np.array):
        if not self.setup:
            self._setup(img)


        # Calculate difference
        img_dif = img - self.last_input
        self.last_input[:] = img[:]

        # Execute a threshold
        img_motion = np.where(np.abs(img_dif) > 20, 255., 0)

        # Calculate motion trace
        output = img_motion + self.decay * self.last_output

        # Clip input to valid values
        output = np.clip(output, 0.0, 255.0)

        # Store output value
        self.last_output = output

        return self._image()

    def _setup(self, img):
        """ For new games setup the correct motion trace.  """
        self.last_input = img.astype(np.float)
        self.last_output = np.zeros_like(self.last_input)
        self.setup = True

    def _image(self):
        """ Return neural network input.  """

        # Stack the last input and output images.
        state = np.dstack([self.last_input, self.last_output])

        # Convert to 8-bit integer.
        # This is done to save space in the replay-memory.
        state = state.astype(np.uint8)

        return state
