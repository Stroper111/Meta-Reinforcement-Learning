import os
import time
import gym
import procgen

from core.agents import AbstractAgent


class BaseAgent(AbstractAgent):
    def __init__(self, setup):
        self.setup = setup
        self.instances = sum(setup.values())

    def run(self):
        pass

    def create_save_directory(self):
        games = '_'.join([f"{game}_{instance}" for game, instance in self.setup.items()])
        # self.save_dir = os.path.join("D:/", "checkpoint", games, self.current_time())
        package_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        save_dir = os.path.join(package_directory, "checkpoint", games, self._current_time())
        return save_dir

    @staticmethod
    def validate_input_gym(setup):
        assert len(setup) == 1, "Only 1 gym environment supported currently."
        valid = [env_spec.id for env_spec in gym.envs.registry.all()]
        valid_keys = '\n\t'.join(valid)
        for game, instance in setup.items():
            assert game in valid, f"Use one of the valid keys:\n\t{valid_keys}"
            assert isinstance(instance, int), "Please only use integers as key values."

    @staticmethod
    def validate_input_procgen(setup):
        valid_games = procgen.env.ENV_NAMES
        for game, instance in setup.items():
            assert game in valid_games, f"Use one of the valid keys: {valid_games}"
            assert isinstance(instance, int), "Please only use integers as key values."

    @staticmethod
    def _current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')
