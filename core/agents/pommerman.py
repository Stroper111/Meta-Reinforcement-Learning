import re

# Pommerman imports

import core.games.pommerman as pommerman

# Tools
from collections import namedtuple

# Core imports (agents can be used to initialize different agents)
from core.games.pommerman import agents
from core.games.pommerman.constants import Result

# To prevent merge conflicts all bots are automatically loaded.
from agents.pommerman import bots

# Agent imports
from core.agents import AbstractAgent
from core.preprocessing.wrappers.pm import *


# using abstract agent enforces same interface
class PommermanAgent(AbstractAgent):
    _step = namedtuple('step', ('states', 'rewards', 'done', 'info'))

    # All these arguments can be overloaded using the kwargs.
    nr_games = 1_000

    render = False
    render_interval = 1
    render_slow_down = True

    my_bot_name: str = 'BaseBot'
    my_bot_index: int = 0
    game_name = 'PommeFFACompetition-v0'

    rewards: dict = dict(kills=0., boxes=0.1, powerups=0., bombs=0.05, alive=0.01)
    wrappers: dict = dict(reset_on_death=False, unpack=False, rgb_array=False, danger_map=False)

    def __init__(self, *args, **kwargs):
        super().__init__(setup=dict())

        # Catch overloading arguments (not useful)
        if args:
            print("Extra arguments:\n\t ", '\n\t '.join(map(str, args)))

        # Catch overloading keyword arguments (replace local variables if available
        if kwargs:
            msg = '\n\t '.join(['%-20s %s' % (key, value) for key, value in kwargs.items()])
            print("Overloading key word arguments:\n\t", msg)

            for key, value in kwargs.items():
                if not hasattr(self, key):
                    raise ValueError(f"Key does not exists, please check: {key}")
                setattr(self, key, value)

        # Summarize available agents (easier swapping of agents).
        self.agents = dict(RandomAgent=agents.RandomAgent,
                           SimpleAgent=agents.SimpleAgent,
                           **self._get_bots)

        # Create environment and store agent location.
        self._env, self.agent_callback = self.create_game(bot_name=self.my_bot_name,
                                                          bot_index=self.my_bot_index,
                                                          game_name=self.game_name)

        self._wrap_game(**self.wrappers, rewards=self.rewards)

    @property
    def _get_bots(self):
        """ Dynamically loads all bots from the bots init file.  """

        my_bots = dict()
        for key in dir(bots):

            # Perform checks
            check_magic_function = re.findall('__.*__', key)
            check_class_instance = isinstance(getattr(bots, key), type)

            if not check_magic_function and check_class_instance:
                my_bots[key] = getattr(bots, key)

        new_bots = '\n\t '.join(['%-20s %s' % (key, value) for key, value in my_bots.items()]) if my_bots else 'None'
        print("\nFound the following bots in file `bots.__init__.py`:\n\t", new_bots)
        return my_bots

    def _wrap_game(self, reset_on_death=False, unpack=False, rewards: dict = None, rgb_array=False, danger_map=False):
        """ Apply wrappers to the game.  """
        if reset_on_death:
            self._env = PommermanResetOnDeath(self._env, agent_idx=self.my_bot_index)

        if unpack:
            self._env = PommermanUnpack(self._env, agent_idx=self.my_bot_index)

        if rgb_array:
            self._env = PommermanRGBObservation(self._env, agent_idx=self.my_bot_index)

        if danger_map:
            self._env = PommermanDangerMap(self._env, agent_idx=self.my_bot_index)

        # Check for a nonzero value in dictionary
        if rewards and sum(1 for value in rewards.values() if value):
            self._env = PommermanReward(self._env, agent_idx=self.my_bot_index, **rewards)

    def create_game(self, bot_name: str = 'BaseBot', bot_index: int = 0, game_name='PommeFFACompetition-v0'):
        """ Create a game with a list of agents. """
        agent_list = [self.agents['BotFreeze']() for _ in range(4)]
        agent_list[bot_index] = self.agents[bot_name]()

        env = pommerman.make(game_name, agent_list)
        return env, agent_list[bot_index]

    def run(self):

        # Bookkeeping
        running_mean = 0.

        # Can be changed to to while loop if necessary.
        for episode in range(1, self.nr_games + 1):

            # Set the starting state and counters.
            done = Result.Incomplete
            states = self._env.reset()
            steps = 0
            reward = 0

            # Keep looping until a single match is over.
            while done == Result.Incomplete:

                # Render the game
                if self.render and episode % self.render_interval == 0:
                    self._env.render(do_sleep=self.render_slow_down)

                # get all actions (this also calls your agents act function.)
                actions = self._env.act(states)

                # Perform a step
                step = self._step(*self._env.step(actions))
                steps += 1

                # Return agent information (if UnpackVec, extra info is obs['obs'])
                if hasattr(self.agent_callback, 'step_results'):
                    self.agent_callback.step_results(step.states[self.my_bot_index],
                                                     step.rewards[self.my_bot_index],
                                                     step.done, step.info)

                # Update info for next step
                states = step.states
                done = step.info['result']
                reward += step.rewards[self.my_bot_index]

            # Bookkeeping
            running_mean = running_mean * 0.99 + reward * 0.01

            # Create a nice print statement
            msg = "\nEpisode {:7,d},\tResult {:4s},\tReward {: 6.2f},\tSteps {:3d}\tRunning mean {: 5.3f}"
            print(msg.format(episode, done.name, reward, steps, running_mean), end='')

            # Give information about epsilon
            if hasattr(self.agent_callback, 'epsilon'):
                print(",\tEpsilon {:6.4f}".format(self.agent_callback.epsilon), end='')

            # If you use a custom reward, this tells the count of the rewards per episode.
            if hasattr(self._env, 'reward_counts'):
                print("\t\t{:s}".format(self._env.reward_counts), end='')

        # Save model if possible
        if hasattr(self.agent_callback, 'save_model'):
            self.agent_callback.save_model(episode=self.nr_games)


if __name__ == '__main__':
    agent = PommermanAgent()
    agent.run()
