import re

import numpy as np

# Pommerman imports
from core.games.pommerman import characters
from core.games.pommerman.agents import BaseAgent

# Core help
from core.memory import BaseMemoryDeque
from core.memory.sampling import BaseSampling

# Local helpers
from agents.pommerman.bots.dqn.simple.model import DQNModel
from agents.pommerman.bots.dqn.simple.data import Commands


class SimpleDQNAgent(BaseAgent):
    # Epsilon control if necessary
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay_rate = 0.999

    # Save settings
    save = True
    save_interval = 1_000
    save_name = "episode {:6,d}.weights"

    # If True will try to load the model.
    resume = True

    def __init__(self, character=characters.Bomber):
        super().__init__(character)

        # Example init
        self.model = DQNModel(input_shape=(11 * 11 * 2,), output_shape=len(Commands.all))
        self.memory = BaseMemoryDeque(size=1_000_000)
        self.sampler = BaseSampling(self.memory, self.model, gamma=0.99, batch_size=64)
        self.commands = Commands()

        # options for saving and loading, when loading make custom name the load directory
        self.model.create_save_directory(agent_name=self.__class__.__name__, game_name='pommerman', custom_name='test')

        # Predefine all variables
        self.old_obs = None
        self.old_action = None
        self.episode = 0
        self.episode_offset = 0

        # If we are resuming, load model and episode offset
        if self.resume:
            self.save_path = self.model.load_checkpoint(load_name='last')
            self.episode_offset = self._locate_int_offset(self.save_path, pattern="[,\d]+", match_idx=0, alternative=0)

            for _ in range(self.episode_offset):
                self._epsilon_decay()

            msg = "\nOffsets calculated:\n\tEpisode: {:6,d}\n\tEpsilon: {:6.4f}"
            print((msg).format(self.episode_offset, self.epsilon))

    @staticmethod
    def _locate_int_offset(string: str, pattern: str, match_idx: int, alternative: int):
        """ Locate offsets inside the save name, if none is found it will be zero (no offsets).  """
        matches = re.findall(pattern, string) if isinstance(string, str) else []
        return int(matches[match_idx].replace(',', '')) if len(matches) > match_idx else alternative

    def act(self, obs, action_space):
        # Main event that is being called on every turn.

        # Create observation
        input = self._model_input(obs)

        # Apply epsilon
        action = self.model.actions_random((1,))
        if np.random.rand() > self.epsilon:
            new_action = self.model.act(input)
            action = self.decide_action(action, new_action, obs['position'], dangermap=obs['danger_map'])

        # Store values for memory
        self.old_obs = input
        self.old_action = action

        return action

    def _model_input(self, obs):
        board = np.expand_dims(obs['board'], axis=0)
        danger_map = np.expand_dims(obs['danger_map'], axis=0)
        combined = np.transpose(np.vstack([board, danger_map]), axes=[1, 2, 0])
        input = np.expand_dims(combined.ravel(), axis=0)
        return input

    def step_results(self, new_obs, reward, done, info):
        """ Function gets called after every step.  """
        input = self._model_input(new_obs)
        self.memory.add(self.old_obs, self.old_action, reward, done, input)

        if len(self.memory) > self.sampler.batch_size * 2:
            self.model.train(*self.sampler.random_batch())

    def episode_end(self, reward):
        """ Function gets called at the end of an episode.  """
        self._epsilon_decay()
        self.episode += 1

        if self.save and self.episode % self.save_interval == 0:
            self.save_model_checkpoint(self.episode)

    def save_model_checkpoint(self, episode):
        """ Helper function to store a checkpoint of a model.  """
        self.model.save_checkpoint(save_name=self.save_name.format(episode + self.episode_offset))

    def save_model(self, episode):
        """ Helpere function to store the model.  """
        self.model.save_model(save_name=self.save_name.format(episode + self.episode_offset))

    def _epsilon_decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)

    def decide_action(self, action: int, new_action: int, my_position: tuple, dangermap: np.ndarray):
        """ To prevent random actions of killing an agent, the actions are only picked if they are safe.  """

        command = self.commands.get_command(action=new_action)
        new_point = command.array + np.array(my_position)

        if min(new_point) < 0 or max(new_point) > 10:
            return action

        if dangermap[tuple(new_point)] > 0:
            return action

        return new_action


if __name__ == '__main__':
    from core.agents import PommermanAgent

    # Example of overloading settings
    settings = dict(
            # Overload used agent.
            my_bot_name=SimpleDQNAgent.__name__,
            my_bot_index=0,

            # Overload game settings
            nr_games=1_000_000,
            render=False,
            render_interval=500,
            render_slow_down=False,

            # Overload wrapper settings
            wrappers=dict(reset_on_death=True, unpack=False, rgb_array=False, danger_map=True),
            rewards=dict(kills=0.5, boxes=0.2, powerups=0.1, bombs=0.05, alive=0.001),
    )

    agent = PommermanAgent(**settings)
    agent.run()
