
import os
import time
import numpy as np

from core.tools import MultiEnv, Scheduler
from core.preprocessing import BasePreProcessing
from core.models import BaseModel
from core.memory.replay_memory import ReplayMemory
from core.memory.sampling import BaseSampling



class BaseAgent:
    def __init__(self):

        self.setup = dict(coinrun=6)
        self.instances = sum(self.setup.values())
        self.env = MultiEnv(self.setup)

        self.save_dir = os.path.join("D:/", "checkpoint", self.current_time())
        self.processor = BasePreProcessing(self.env, save_dir=self.save_dir)

        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.action_space = self.processor.output_shape()

        self.model = BaseModel(self.input_shape, self.action_space)

        self.kwargs = dict(time_update=5)
        self.scheduler = Scheduler(self.env, **self.kwargs)

    def run(self):
        images = self.scheduler.reset_images
        for env in self.scheduler:
            actions = np.random.randint(0, self.action_space, self.instances)
            images, rewards, dones, infos = env.step(actions)


    @staticmethod
    def current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')