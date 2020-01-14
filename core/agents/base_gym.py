import numpy as np
import gym

from core.agents import BaseAgent
from core.tools import Scheduler
from core.models import BaseModelGym
from core.memory.base_replay_memory import BaseReplayMemory
from core.memory.sampling import BaseSamplingGym
from core.preprocessing import BasePreProcessingGym

from collections import deque


class BaseAgentGym(BaseAgent):
    def __init__(self, setup):
        super().__init__(setup)

        self.setup = setup
        self.instances = sum(setup.values())
        self.env = self._create_env(setup)

        self.save_dir = self.create_save_directory()

        self.processor = BasePreProcessingGym(self.env, save_dir=self.save_dir, history_size=5)

        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.action_space = self.processor.output_shape()

        self.model = BaseModelGym(self.input_shape, self.action_space)
        # self.model.load_checkpoint(self.save_dir)
        self.memory = BaseReplayMemory(size=10_000, shape=self.input_shape, action_space=self.action_space)
        self.sampler = BaseSamplingGym(self.memory, self.model, gamma=0.95, batch_size=512)
        self.loss = deque([0], maxlen=100)

        kwargs = dict(episode_limit=1_000, time_update=5)
        self.scheduler = Scheduler(self.env, **kwargs)

        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def run(self):
        state = self.scheduler.reset_images
        q_values, action = self.model.actions(state['rgb'])

        for env, update, episode, steps in self.scheduler:
            # Remember all things are stuck in an array
            state, reward, done, info = env.step(action)

            self.memory.add(state=state['rgb'][0], action=action[0], reward=reward[0], end_episode=done[0])

            if steps > self.sampler.batch_size:
                self.loss.append(self.model.train_once(sampling=self.sampler))

            if self.memory.is_full():
                self.memory.refill_memory()

            if self.memory.pointer > self.sampler.batch_size:
                self.loss.append(self.model.train_once(self.sampler))

            if update:
                self.model.save_checkpoint(self.save_dir, episode, steps * self.instances)
                loss_msg = '{:15,.4f}'.format(np.mean(self.loss))
                print('\r\tloss (average)'.ljust(18), loss_msg)

            q_values, action = self.model.actions(state['rgb'])

            # Handle exploration
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0, self.action_space, (1,))
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        print("\nRun completed, models and logs are located here:\n", self.save_dir.replace("\\", "/"))

    def _create_env(self, setup):
        self.validate_game_input(setup, gym_env=True)
        for game, instances in setup.items():
            return gym.make(game)
