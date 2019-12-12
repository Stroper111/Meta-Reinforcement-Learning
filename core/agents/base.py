import os
import time
import numpy as np

from collections import deque

from core.tools import MultiEnv, Scheduler
from core.preprocessing import BasePreProcessing
from core.models import BaseModel
from core.memory.replay_memory import ReplayMemory
from core.memory.sampling import BaseSampling


class BaseAgent:
    def __init__(self):
        self.setup = dict(bigfish=1)
        self.instances = sum(self.setup.values())
        self.env = MultiEnv(self.setup)

        self.save_dir = os.path.join("D:/", "checkpoint", self.current_time())
        self.processor = BasePreProcessing(self.env, save_dir=self.save_dir, history_size=30)

        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.action_space = self.processor.output_shape()

        self.model = BaseModel(self.input_shape, self.action_space)
        self.memories = self._create_memories()
        self.samplers = self._create_samplers()
        self.loss = self._create_loss()

        self.kwargs = dict(step_update=5_000)
        self.scheduler = Scheduler(self.env, **self.kwargs)

    def _create_memories(self):
        memories = []
        for _ in range(self.instances):
            memories.append(ReplayMemory(size=200_000, shape=self.input_shape, action_space=self.action_space))
        return memories

    def _create_samplers(self):
        samplers = []
        for k in range(self.instances):
            samplers.append(BaseSampling(self.memories[k], batch_size=128))
        return samplers

    def _create_loss(self):
        loss = []
        for _ in range(self.instances):
            loss.append(deque([0], maxlen=100))
        return loss

    def run(self):
        images = self.scheduler.reset_images
        actions = np.argmax(self.model.predict(self.reformat_states(images)), axis=1)

        for env, update in self.scheduler:
            images, rewards, dones, infos = env.step(actions)
            q_values = self.model.predict(self.reformat_states(images))

            for k in range(self.instances):
                self.memories[k].add(state=images['rgb'][k], q_values=q_values[k], action=actions[k],
                                     reward=rewards[k], end_episode=dones[k])

            if update:
                for k in range(self.instances):
                    if self.memories[k].pointer_ratio() >= 0.1:
                        self.memories[k].update()
                        self.loss[k].append(np.mean(self.model.train(sampling=self.samplers[k])))
                        print('\tloss'.ljust(17), ''.join(['{:15,.4f}'.format(np.mean(game)) for game in self.loss]))

            actions = np.argmax(q_values, axis=1)

    def reformat_states(self, states):
        """  Transforms the input of  stacked frame to the required format for the model.  """
        return np.array(states['rgb']).transpose([0, 2, 3, 1])

    @staticmethod
    def current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')
