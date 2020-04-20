import re

# Pommerman imports
from core.games.pommerman import characters
from core.games.pommerman.constants import Action
from core.games.pommerman.agents import BaseAgent

# Core help
from core.memory import BaseMemoryDeque
from core.memory.sampling import BaseSampling


class BaseBot(BaseAgent):
    # The output shape (in correct order)
    actions = [Action.Stop, Action.Up, Action.Down, Action.Left, Action.Right, Action.Bomb]

    # Epsilon control if necessary
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay_rate = 0.999

    # Save settings
    save = False
    save_interval = 0
    save_name = "episode {:6,d}"

    # If True will try to load the model.
    resume = False

    def __init__(self, character=characters.Bomber):
        super().__init__(character)

        # Example init
        self.model = None
        self.memory = BaseMemoryDeque(size=1_000_000)
        self.sampler = BaseSampling(self.memory, self.model, gamma=0.95, batch_size=20)

        # options for saving and loading, when loading make custom name the load directory
        self.model.create_save_directory(agent_name=self.__class__.__name__, game_name='pommerman', custom_name=None)

        if self.resume:
            self.save_path = self.model.load_checkpoint(load_name='last')
            self.episode_offset = self._locate_int_offset(self.save_path, pattern="\d+", match_idx=0, alternative=0)

    @staticmethod
    def _locate_int_offset(string: str, pattern: str, match_idx: int, alternative: int):
        """ Locate offsets inside the save name, if none is found it will be zero (no offsets).  """
        matches = re.findall(pattern, string)
        return int(matches[match_idx]) if len(matches) > match_idx else alternative

    def act(self, obs, action_space):
        # Main event that is being called on every turn.
        return Action.Stop

    def step_results(self, obs, reward, done, info):
        """ Function gets called after every step.  """
        pass

    def episode_end(self, reward):
        """ Function gets called at the end of an episode.  """
        pass

    def save_model_checkpoint(self, episode):
        """ Helper function to store a checkpoint of a model.  """
        self.model.save_checkpoint(save_name=self.save_name.format(episode + self.episode_offset))

    def save_model(self, episode):
        """ Helpere function to store the model.  """
        self.model.save_model(save_name=self.save_name.format(episode + self.episode_offset))

    def _epsilon_decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_min)


if __name__ == '__main__':
    from core.agents import PommermanAgent

    # Example of overloading settings
    settings = dict(
            # Overload used agent.
            # my_bot_name=BaseBot.__name__,
            my_bot_name='RandomAgent',
            my_bot_index=1,

            # Overload game settings
            nr_games=2,
            render=True,
            render_interval=1,
            render_slow_down=True,

            # Overload wrapper settings
            wrappers=dict(reset_on_death=True, unpack=False, rgb_array=False, danger_map=False),
            # rewards=dict(kills=0., boxes=0.1, powerups=0., bombs=0.05, alive=0.01),
            rewards=None,
    )

    agent = PommermanAgent(**settings)
    agent.run()
