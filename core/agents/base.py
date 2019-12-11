
import os
import time

from core.tools import MultiEnv
from core.preprocessing import BasePreProcessing
from core.models import BaseModel
from core.memory.replay_memory import ReplayMemory
from core.memory.sampling import BaseSampling


class BaseAgent:
    def __init__(self):

        self.setup = dict(cionrun=1)
        self.env = MultiEnv(self.setup)

        self.save_dir = os.path.join("D:/", "checkpoint", self.current_time())
        self.processor = BasePreProcessing(self.env, save_dir=self.save_dir)

        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.output_shape = self.processor.output_shape()

        self.model = BaseModel(self.input_shape, self.output_shape)

    def run(self):
        while True:



            pass

    @staticmethod
    def current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')