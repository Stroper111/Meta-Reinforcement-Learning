import os
import random
import numpy as np
import re

# Core helpers
from agents.pommerman.bots import BaseBot
from core.games.pommerman import characters

# Local imports (full import due to main)
from agents.pommerman.bots.pg.simple.model import PommermanPGModel
from agents.pommerman.bots.pg.simple.data import Commands

# Seed for reproducibility (this is not guaranteed using Keras/PyTorch)
seed = 1234
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


class SimplePGAgent(BaseBot):
    input_shape = (11 * 11 * 2,)
    output_shape = 6
    hidden_layers = [200, output_shape]
    initialization = 'xavier'

    learning_rate = 1e-3
    gamma = 0.99
    decay_rate = 0.99
    batch_size = 10

    resume = False
    save = False
    save_interval = 1
    save_name = 'episode {:6d}, mean {: 8.5f}.pkl'

    def __init__(self, character=characters.Bomber):
        super().__init__(character)

        # Define gradient model
        self.model = PommermanPGModel(self.input_shape, self.output_shape, self.hidden_layers, self.initialization)

        # options for saving and loading, when loading make custom name the load directory
        self.model.create_save_directory(agent_name=self.__class__.__name__, game_name='pommerman', custom_name='test')

        # Predefine all variables
        self.episode = 0
        self.episode_offset = 0
        self.running_mean = 0
        self.commands = Commands()

        # This is our memory that we will use later on.
        self.obs_, self.actions_, self.hidden_, self.dlogps_, self.discounted_ = [], [], [], [], []

        # If we are resuming, load model and episode offset
        if self.resume:
            self._resume()

    def _resume(self):
        """ Perform all actions that are required for resuming a match.  """
        self.save_path = self.model.load_checkpoint(load_name='last')
        self.episode_offset = self._locate_int_offset(self.save_path, pattern="[,\d]+", match_idx=0, alternative=0)
        self.running_mean = self._locate_int_offset(self.save_path, pattern="[,\d]+", match_idx=1, alternative=0)

        # Apply epsilon decay
        for _ in range(self.episode_offset):
            self._epsilon_decay()

        # Give back information
        msg = "Offsets calculated:\n\tEpisode: {:6,d}\n\tEpsilon: {:6.4f}\n\tMean: {: 6.4f}\n"
        print((msg).format(self.episode_offset, self.epsilon, self.running_mean))

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
        probabilities, hidden = self.model.policy_forward(input)

        if np.random.rand() < self.epsilon:
            # Calculate new action
            new_action = int(np.argmax(probabilities))
            action = self.decide_action(action, new_action, obs['position'], dangermap=obs['danger_map'])

        # Store variables before taking a step
        self.obs_.append(input)
        self.hidden_.append(hidden)
        self.dlogps_.append(probabilities)
        self.actions_.append(int(action))

        # Store values for memory
        self.old_obs = input
        self.old_action = action
        return action

    def _model_input(self, obs):
        board = np.expand_dims(obs['board'], axis=0)
        danger_map = np.expand_dims(obs['danger_map'], axis=0)
        combined = np.transpose(np.vstack([board, danger_map]), axes=[1, 2, 0])
        input = combined.ravel()
        return input

    def step_results(self, new_obs, reward, done, info):
        """ Function gets called after every step.  """

        # Store reward for previous actions
        self.discounted_.append(reward)

    def episode_end(self, reward):
        """ Function gets called at the end of an episode.  """

        # Bookkeeping
        self._epsilon_decay()
        self.episode += 1
        self.running_mean = self.running_mean * 0.99 + reward * 0.01

        # Calculating all gradients
        episode_obs = np.array(np.vstack(self.obs_), dtype=np.float64)
        episode_hidden = np.array(np.vstack(self.hidden_), dtype=np.float64)
        episode_logps = np.array(np.vstack(self.dlogps_), dtype=np.float64)
        episode_rewards = np.array(np.vstack(self.discounted_), dtype=np.float64)

        # compute the discounted reward backwards through time
        discounted_episode_reward = self.model.discount_rewards(episode_rewards, gamma=self.gamma)

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_episode_reward -= np.mean(discounted_episode_reward)
        discounted_episode_reward /= np.std(discounted_episode_reward)

        # modulate the gradient with advantage (policy gradient magic happens right here.)
        episode_logps[[np.arange(len(episode_logps))], [self.actions_]][0] *= discounted_episode_reward[:, 0]

        # Calculate all the backward gradients and store them in the layers
        self.model.policy_backward(episode_hidden, episode_logps, episode_obs)

        if self.episode % self.batch_size == 0:
            self.model.apply_gradients(decay_rate=self.decay_rate, learning_rate=self.learning_rate)

        if self.save and self.episode % self.save_interval == 0:
            self.save_model_checkpoint(self.episode)

        # Clear memory
        self.obs_, self.actions_, self.hidden_, self.dlogps_, self.discounted_ = [], [], [], [], []

    def save_model_checkpoint(self, episode):
        """ Helper function to store a checkpoint of a model.  """
        self.model.save_checkpoint(save_name=self.save_name.format(episode + self.episode_offset, self.running_mean))

    def save_model(self, episode):
        """ Helpere function to store the model.  """
        self.model.save_model(save_name=self.save_name.format(episode + self.episode_offset, self.running_mean))

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
            my_bot_name=SimplePGAgent.__name__,
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
