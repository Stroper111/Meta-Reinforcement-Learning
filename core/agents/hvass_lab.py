import numpy as np
import gym

from copy import deepcopy

from core.agents import BaseAgent
from core.tools import Scheduler, LinearControlSignal
from core.preprocessing import PreProcessingHvasslab
from core.models import HvassLab as ModelHvassLab
from core.memory import ReplayMemoryHvassLab


class HvassLabAgent(BaseAgent):
    def __init__(self, setup: dict, training=True):
        super().__init__(setup)

        self.setup = setup
        self.training = training

        self.instances = sum(self.setup.values())
        self.env = self._create_env(setup)

        self.save_dir = self.create_save_directory()

        kwargs = dict(gym=True, rescaling_dim=(105, 80), motion_tracer=True, save_dir=self.save_dir, history_size=20)
        self.processor = PreProcessingHvasslab(self.env, **kwargs)

        # Get the processed environment and calculate the input and output shape.
        self.env = self.processor.env
        self.input_shape = self.processor.input_shape()
        self.action_space = self.processor.output_shape()

        self.model = ModelHvassLab(self.input_shape, self.action_space)
        # self.model.load_checkpoint(self.save_dir)

        # Controller for setting time, step and episodes limits and updates.
        kwargs = dict(time_update=10)
        self.scheduler = Scheduler(self.env, **kwargs)

        # Initialize the Replay Memory and set all control signals
        if self.training:
            self.memory = ReplayMemoryHvassLab(size=1_000_000, shape=self.input_shape, action_space=self.action_space,
                                               stackedframes=False)

            self.learning_rate_control = LinearControlSignal(
                start_value=1e-3, end_value=1e-5, num_iterations=5e6)

            self.loss_limit_control = LinearControlSignal(
                start_value=0.1, end_value=0.015, num_iterations=5e6)

            self.max_epochs_control = LinearControlSignal(
                start_value=5.0, end_value=10.0, num_iterations=5e6)

            self.replay_fraction = LinearControlSignal(
                start_value=0.1, end_value=1.0, num_iterations=5e6)
        else:
            self.learning_rate_control = None
            self.loss_limit_control = None
            self.max_epochs_control = None
            self.replay_fraction = None
            self.memory = None

    def run(self):
        images = self.scheduler.reset_images
        q_values, actions = self.model.actions(self.reformat_states(images))

        for env, update, episode, steps in self.scheduler:
            images, rewards, dones, infos = env.step(actions)
            # Please always use deepcopy for this, due to memory usage, otherwise all layzframes are stored unpacked
            q_values, actions_new = self.model.actions(self.reformat_states(deepcopy(images)))

            if self.training:
                self.memory.add(state=images['rgb'][0], q_values=q_values, action=actions[0],
                                reward=rewards[0], end_episode=dones[0])

                use_fraction = self.replay_fraction.get_value(iteration=steps)
                if self.memory.is_full() or self.memory.pointer_ratio() >= use_fraction:
                    self.memory.update()

                    # TODO add logging of Q-values
                    learning_rate = self.learning_rate_control.get_value(iteration=steps)
                    loss_limit = self.loss_limit_control.get_value(iteration=steps)
                    max_epochs = self.max_epochs_control.get_value(iteration=steps)

                    self.model.change_learning_rate(learning_rate)
                    self.model.optimize(replay_memory=self.memory,
                                        learning_rate=learning_rate,
                                        loss_limit=loss_limit,
                                        max_epochs=max_epochs)

                    self.model.save_checkpoint(self.save_dir, episode, steps * self.instances)
                    self.memory.reset()

            actions = actions_new

    @staticmethod
    def reformat_states(states):
        """  Transforms the input of stacked frame to the required format for the model.  """
        return np.array(states['rgb'])

    def _create_env(self, setup):
        self.validate_game_input(setup, gym_env=True)
        return gym.make(list(setup.keys())[0])
