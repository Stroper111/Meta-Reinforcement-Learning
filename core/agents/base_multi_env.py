import numpy as np

from collections import deque
from copy import deepcopy

from core.agents import BaseAgent
from core.tools import MultiEnv, Scheduler
from core.preprocessing import BasePreProcessingMultiEnv
from core.models import BaseModel
from core.memory.base_replay_memory import BaseReplayMemory
from core.memory.sampling import BaseSamplingMultiEnv


class BaseAgentMultiEnv(BaseAgent):
    def __init__(self, setup: list):
        super().__init__(setup)

        self.setup = self._convert_setup_to_dict(setup)
        self.instances = sum(self.setup.values())
        self.env = MultiEnv(self.setup)

        self.save_dir = self.create_save_directory()
        self.processor = BasePreProcessingMultiEnv(self.env, save_dir=self.save_dir, history_size=50)

        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.action_space = self.processor.output_shape()

        self.model = BaseModel(self.input_shape, self.action_space)
        # self.model.load_checkpoint(self.save_dir)

        self.memories = self._create_memories()
        self.samplers = self._create_samplers()
        self.loss = self._create_loss()

        self.kwargs = dict(time_update=10)
        self.scheduler = Scheduler(self.env, **self.kwargs)

        self.replay_factor = 0.1

    def _create_memories(self):
        memories = []
        for _ in range(self.instances):
            memories.append(BaseReplayMemory(size=50_000, shape=self.input_shape, action_space=self.action_space,
                                             stacked_frames=True))
        return memories

    def _create_samplers(self):
        samplers = []
        for k in range(self.instances):
            samplers.append(BaseSamplingMultiEnv(self.memories[k], batch_size=128))
        return samplers

    def _create_loss(self):
        loss = []
        for _ in range(self.instances):
            loss.append(deque([0], maxlen=100))
        return loss

    def run(self):
        images = self.scheduler.reset_images
        q_values, actions = self.model.actions(self.reformat_states(images))

        for env, update, episode, steps in self.scheduler:
            images, rewards, dones, infos = env.step(actions)
            # Please always use deepcopy for this, since you use a lot of memory otherwise (you unpack all layzframes)
            q_values, actions_new = self.model.actions(self.reformat_states(deepcopy(images)))

            for k in range(self.instances):
                if self.memories[k].is_full():
                    self.memories[k].refill_memory()

                self.memories[k].add(state=images['rgb'][k], action=actions[k], reward=rewards[k], end_episode=dones[k])

            for k in range(self.instances):
                if self.memories[k].pointer_ratio() >= self.samplers[k].batch_size:
                    self.loss[k].append(self.model.train_once(sampling=self.samplers[k]))
                    # self.model.save_checkpoint(self.save_dir, episode, steps * self.instances)

            if update:
                loss_msg = ''.join(['{:15,.4f}'.format(np.mean(game)) for game in self.loss])
                print('\r\tloss (average)'.ljust(18), loss_msg)

            actions = actions_new

    @staticmethod
    def reformat_states(states):
        """  Transforms the input of stacked frame to the required format for the model.  """
        return np.array(states['rgb']).transpose([0, 2, 3, 1])
