import os
import numpy as np
import time
import gym

from core.tools import Scheduler
from core.models import BaseModelGym
from core.memory.replay_memory import ReplayMemory
from core.memory.sampling import BaseSamplingGym
from core.preprocessing import BasePreProcessingGym

from collections import deque



class BaseAgentGym:
    def __init__(self, setup):
        self.setup = setup
        self.instances = sum(setup.values())
        self.env = self._create_env(setup)

        games = '_'.join([f"{game}_{instance}" for game, instance in self.setup.items()])
        self.save_dir = os.path.join("D:/", "checkpoint", games, self.current_time())
        self.processor = BasePreProcessingGym(self.env, save_dir=self.save_dir, history_size=5)

        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.action_space = self.processor.output_shape()

        self.model = BaseModelGym(self.input_shape, self.action_space)
        # self.model.load_checkpoint(self.save_dir)
        self.memory = [ReplayMemory(size=2_000, shape=self.input_shape, action_space=self.action_space)]
        self.sampler = [BaseSamplingGym(self.memory[0], self.model, gamma=0.95, batch_size=32)]
        self.loss = [deque([0], maxlen=100)]

        kwargs = dict(episode_limit=1_000, episode_update=1)
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

            for k in range(self.instances):
                memory = self.memory[k]
                memory.add(state=state['rgb'][k], action=action[k], reward=reward[k], end_episode=done[k])

                if steps > self.sampler[k].batch_size:
                    self.loss[k].append(self.model.train_once(sampling=self.sampler[k]))

                if memory.is_full():
                    memory.refill_memory()

            if update:
                self.model.save_checkpoint(self.save_dir, episode, steps * self.instances)
                loss_msg = ''.join(['{:15,.4f}'.format(np.mean(game)) for game in self.loss])
                print('\r\tloss (average)'.ljust(18), loss_msg)

            q_values, action = self.model.actions(state['rgb'])

            if (np.random.rand() < self.epsilon):
                action = np.random.randint(0, self.action_space, (1,))
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        print("\nRun completed, models and logs are located here:\n", self.save_dir.replace("\\", "/"))

    def _create_env(self, setup):
        game = self._validate_input(setup)
        return gym.make(game)

    @staticmethod
    def _validate_input(setup):
        assert len(setup) == 1, "Only 1 gym environment supported currently."
        valid = [env_spec.id for env_spec in gym.envs.registry.all()]
        valid_keys = '\n\t'.join(valid)
        for game, instance in setup.items():
            assert game in valid, f"Use one of the valid keys:\n\t{valid_keys}"
            assert isinstance(instance, int), "Please only use integers as key values."
            return game

    @staticmethod
    def current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')
