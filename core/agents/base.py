import os
import time
import gym
import procgen

from core.agents import AbstractAgent


class BaseAgent(AbstractAgent):


    def __init__(self, setup: list):
        self.setup = self._convert_setup_to_dict(setup)
        self.instances = sum(self.setup.values())

    def run(self):
        pass

    def create_save_directory(self):
        games = '_'.join([f"{game}_{instance}" for game, instance in self.setup.items()])
        # self.save_dir = os.path.join("D:/", "checkpoint", games, self.current_time())
        package_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        save_dir = os.path.join(package_directory, "checkpoint", games, self._current_time())
        return save_dir

    @staticmethod
    def validate_game_input(setup, gym_env=False):
        if gym_env:
            valid_games = [env_spec.id for env_spec in gym.envs.registry.all()]
            assert len(setup) == 1, "Only 1 gym environment supported currently."
        else:
            valid_games = procgen.env.ENV_NAMES

        valid_keys = '\n\t'.join(valid_games)
        for game, instance in setup.items():
            assert game in valid_games, f"Use one of the valid keys:\n\t{valid_keys}"
            assert isinstance(instance, int), "Please only use integers as key values."

    @staticmethod
    def _current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')

    def _convert_setup_to_dict(self, setup_old: [dict, list]):
        if isinstance(setup_old, dict):
            return setup_old

        if isinstance(setup_old, list):
            if isinstance(setup_old[0], list):
                setup_old = [info for each in setup_old for info in each]

            key = None
            setup_new = dict()
            for idx, each in enumerate(setup_old):
                if each.isdigit():
                    setup_new[key] = int(each)
                else:
                    key = each
                    setup_new[key] = 1
            return setup_new

        raise ValueError("Setup is not in a valid format, either use a dictionary or list.")