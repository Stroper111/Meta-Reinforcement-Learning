import numpy as np
import gym

from copy import deepcopy

from core.agents import BaseAgent
from core.tools import Scheduler, LinearControlSignal, EpsilonGreedy
from core.preprocessing import PreProcessingHvasslab
from core.models import HvassLab as ModelHvassLab
from core.memory import ReplayMemoryHvassLab


class HvassLabAgent(BaseAgent):
    def __init__(self, setup: [dict, list], training=True):
        super().__init__(setup)

        self.setup = self._convert_setup_to_dict(setup)
        self.training = training

        self.instances = sum(self.setup.values())
        self.env = self._create_env(self.setup)

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
        kwargs = dict(step_limit=50_000_000, write_summary=False)
        self.scheduler = Scheduler(self.env, **kwargs, )

        # Controller for exploration vs exploitation
        self.epsilon_greedy = EpsilonGreedy(start_value=1.0,
                                            end_value=0.05,
                                            num_iterations=1e6,
                                            num_actions=self.action_space,
                                            epsilon_testing=0.01)

        # Initialize the Replay Memory and set all control signals
        if self.training:
            self.memory = ReplayMemoryHvassLab(size=200_000, shape=self.input_shape, action_space=self.action_space,
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
        q_values = self.model.actions(self.reformat_states(images))

        num_lives = 5
        for env, update, episode, steps in self.scheduler:

            # Determine the action that the agent must take in the game-environment.
            # The epsilon is just used for printing further below.
            actions, epsilon = self.epsilon_greedy.get_action(q_values=q_values,
                                                              iteration=steps,
                                                              training=self.training)

            images, rewards, dones, infos = env.step(actions)
            # Please always use deepcopy for this, due to memory usage, otherwise all layzframes are stored unpacked
            q_values = self.model.actions(self.reformat_states(deepcopy(images)))

            num_lives_new = self.get_lives()
            end_life = (num_lives_new < num_lives)
            num_lives = num_lives_new

            if self.training:
                self.memory.add(state=images['rgb'][0], q_values=q_values, action=actions[0],
                                reward=rewards[0], end_life=end_life, end_episode=dones[0])

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

                if self.training and dones[0]:
                    summary = env.last_episode_info()
                    msg = "{episode:6,d}:{states:11,d}\t Epsilon: {epsilon:4.2f}\t" \
                          " Reward: {reward_episode:.1f}\t Episode Mean: {reward_mean:.1f}"
                    print(msg.format(episode=episode, states=steps,
                                     epsilon=epsilon, reward_episode=summary['reward'],
                                     reward_mean=summary['mean']))

    def get_lives(self):
        """ Get the number of lives the agent has in the game-environment.  """
        return self.env.unwrapped.ale.lives()

    @staticmethod
    def reformat_states(states):
        """  Transforms the input of stacked frame to the required format for the model.  """
        return np.array(states['rgb'])

    def _create_env(self, setup):
        self.validate_game_input(setup, gym_env=True)
        return gym.make(list(setup.keys())[0])
