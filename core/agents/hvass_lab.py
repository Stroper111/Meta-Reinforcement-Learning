import numpy as np
import gym

from collections import deque
from copy import deepcopy

from core.agents import BaseAgent
from core.tools import Scheduler
from core.preprocessing import BasePreProcessingGym
from core.models import HvassLab as HvassLabModel
from core.memory import ReplayMemoryHvassLab
from core.memory.sampling import BaseSamplingGym


class HvassLabAgent(BaseAgent):
    def __init__(self, setup: dict):
        super().__init__(setup)

        self.setup = setup
        self.instances = sum(self.setup.values())
        self.env = self._create_env(setup)

        self.save_dir = self.create_save_directory()
        self.processor = BasePreProcessingGym(self.env, save_dir=self.save_dir, history_size=5)

        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.action_space = self.processor.output_shape()

        self.model = HvassLabModel(self.input_shape, self.action_space)
        # self.model.load_checkpoint(self.save_dir)

        self.memory = ReplayMemoryHvassLab(size=50_000, shape=self.input_shape, action_space=self.action_space,
                                           stackedframes=True)
        self.samplers = BaseSamplingGym(replay_memory=self.memory, model=self.model, gamma=0.95, batch_size=128)
        self.loss = deque([0], maxlen=100)

        self.kwargs = dict(time_update=10)
        self.scheduler = Scheduler(self.env, **self.kwargs)

        # TODO add control signals
        self.replay_factor = 0.1

    def run(self):
        images = self.scheduler.reset_images
        q_values, actions = self.model.actions(self.reformat_states(images))

        for env, update, episode, steps in self.scheduler:
            images, rewards, dones, infos = env.step(actions)
            # Please always use deepcopy for this, due to memory usage, otherwise all layzframes are stored unpacked
            q_values, actions_new = self.model.actions(self.reformat_states(deepcopy(images)))

            if self.memory.is_full():
                self.memory.refill_memory()

            self.memory.add(state=images['rgb'][0], q_values=q_values, action=actions[0],
                            reward=rewards[0], end_episode=dones[0])

            if update:
                self.model.save_checkpoint(self.save_dir, episode, steps * self.instances)
                loss_msg = '{:15,.4f}'.format(np.mean(self.loss))
                print('\r\tloss (average)'.ljust(18), loss_msg)

            actions = actions_new

    @staticmethod
    def reformat_states(states):
        """  Transforms the input of stacked frame to the required format for the model.  """
        return np.array(states['rgb'])

    def _create_env(self, setup):
        self.validate_game_input(setup, gym_env=True)
        return gym.make(list(setup.keys())[0])
