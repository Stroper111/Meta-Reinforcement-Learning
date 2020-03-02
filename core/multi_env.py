import numpy as np
import gym
import procgen
import difflib
import warnings

from collections import namedtuple, deque

from core.preprocessing.wrappers.envs import GymWrapper, GymWrapperMP
from core.preprocessing.wrappers import UnpackRGB


class MultiEnv:
    """
        This class controls multiple different game environments.
        It can be used to create an agent that can learn from
        several different environments at the same time.

        setup: dict
            A dictionary where the key is the game name as represented
            in procgen.env.ENV_NAMES or gym, and the value is the number of
            instances of this game.
        use_multiprocessing: bool
            Only has effect when there is a single gym game with multiple instances.
            If True will create multiple processes to run games simultaneously.

    """
    _step = namedtuple("step", ("img", "reward", "done", "info"))

    def __init__(self, setup: dict, use_multiprocessing: bool = False):
        self.setup = setup
        self.use_multiprocessing = use_multiprocessing

        self.instances = sum(setup.values())
        self.venv = self._setup_env(setup)

    def __getattr__(self, item):
        """ Back up for invoking an attribute not overruled here.  """
        data = dict()
        for game, venv in zip(self.setup, self.venv):
            if hasattr(venv, item):
                data[game] = getattr(venv, item)
            else:
                raise AttributeError(f"{venv.__class__.__name__} doesn't have attribute '{item}'")
        return data

    @staticmethod
    def create_games_procgen(setup):
        """ Helper for creating all environments with the correct number of instances.  """
        venv = []
        for game, instances in setup.items():
            env = procgen.ProcgenEnv(num_envs=instances, env_name=game, distribution_mode='easy')
            venv.append(UnpackRGB(env))
        return venv

    @staticmethod
    def create_games_gym(setup, use_multiprocess: bool = False):
        """ Helper for creating all environments with the correct number of instances.  """
        venv = []
        if use_multiprocess:
            for game, instances in setup.items():
                venv.append(GymWrapperMP(game, instances))
        else:
            for game, instances in setup.items():
                venv.append(GymWrapper(game, instances))
        return venv

    def reset(self):
        """ in the new gym environment reset is the same as creating a new gym instance.  """
        images = [venv.reset() for venv in self.venv]
        images = np.vstack(images)
        return images

    def step(self, actions):
        """ Performs a step in all environments and combines all results.  """
        results = list()
        actions_processed = 0

        for env in self.venv:
            step_input = actions[actions_processed:actions_processed + env.instances]
            results.append(self._step(*env.step(step_input)))

        # Zip is it owns inverse.
        results = self._step(*zip(*results))

        img = np.vstack(results.img)
        reward = np.hstack(results.reward)
        done = np.hstack(results.done)
        info = np.hstack(results.info)
        results = self._step(img, reward, done, info)
        return results

    def render(self):
        """ Render all environments in their own window.  """
        [venv.render() for venv in self.venv]

    def close(self):
        """ Close all the environments.  """
        [venv.close() for venv in self.venv]
        self.venv = None

    @property
    def action_space(self):
        """ Return the action space for every game as a dictionary with the key as game name.  """
        data = dict()
        for game, venv in zip(self.setup, self.venv):
            data[game] = venv.action_space
            if isinstance(venv.action_space, list):  # Convert gym to procgen action space
                data[game] = next(iter(venv.action_space))
        return data

    def _setup_env(self, setup):
        games = dict(gym=dict(), procgen=dict())
        ids_gym = [each for each in gym.envs.registry.env_specs.keys()]
        ids_procgen = procgen.env.ENV_NAMES

        for game, instance in setup.items():
            # Add games to procgen
            if game in ids_procgen:
                games['procgen'][game] = instance

            # Add games to gym
            elif game in ids_gym:
                games['gym'][game] = instance

            else:
                raise ValueError(f"The game `{game}` is not in gym or procgen. "
                                 f"Closest matches: {difflib.get_close_matches(game, ids_gym + ids_procgen)}")

        if len(games['gym']) > 1:
            warnings.warn("Multiple gym games, make sure image space and actions will be equal!", UserWarning)

        if games['gym'] and games['procgen']:
            warnings.warn("A mix of gym and procgen, make sure image space and actions will be equal!", UserWarning)

        venv = deque()
        venv.extend(self.create_games_procgen(games["procgen"]))
        venv.extend(self.create_games_gym(games["gym"], self.use_multiprocessing))
        return venv
