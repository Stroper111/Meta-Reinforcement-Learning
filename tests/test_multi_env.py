import numpy as np
import unittest

from core import MultiEnv


class TestMultiEnv(unittest.TestCase):

    def test_gym_single(self):
        """ testing different single gym games consisting of varying observation space.  """
        setup_1d = {"Taxi-v3": 1}
        setup_2d = {"CartPole-v0": 1}
        setup_3d = {"MsPacman-v0": 1}

        for setup in [setup_1d, setup_2d, setup_3d]:
            self._game_step(setup)
            self._game_loop(setup, nr_games=2)

    def test_gym_multiple(self):
        """ testing different multiple gym games consisting of varying observation space.  """
        setup_1d = {"Taxi-v3": 3}
        setup_2d = {"CartPole-v0": 3}
        setup_3d = {"MsPacman-v0": 3}

        for setup in [setup_1d, setup_2d, setup_3d]:
            self._game_step(setup)
            self._game_loop(setup, nr_games=2)

    def test_procgen_single(self):
        """ Testing single game instances of procgen.  """
        setup_1 = dict(coinrun=1)
        setup_2 = dict(bigfish=1)

        for setup in [setup_1, setup_2]:
            self._game_step(setup)
            self._game_loop(setup, nr_games=2)

    def test_procgen_multiple(self):
        """ Testing single game instances of procgen.  """
        setup_1 = dict(coinrun=3)
        setup_2 = dict(bigfish=3)

        for setup in [setup_1, setup_2]:
            self._game_step(setup)
            self._game_loop(setup, nr_games=2)

    def test_mixed(self):
        """ Testing combination of games (separately gym and procgen).  """
        setup_1 = {'Pong-v0': 1, 'SpaceInvaders-v0': 1}
        setup_2 = dict(bigfish=1, dodgeball=1)

        for setup in [setup_1, setup_2]:
            self._game_step(setup)
            self._game_loop(setup, nr_games=2)

    def test_multiprocessing(self):
        """ Testing of multiprocessing.  """
        setup_1 = {"MsPacman-v0": 3}
        setup_2 = dict(bigfish=3)

        setup_3 = {'Pong-v0': 3, 'SpaceInvaders-v0': 3}
        setup_4 = dict(bigfish=3, dodgeball=3)

        for setup in [setup_1, setup_2, setup_3, setup_4]:
            self._game_step(setup)
            # self._game_loop(setup, nr_games=2, use_mp=True)

    def _game_step(self, setup):
        """ Checks all game steps on normal rewards.  """
        venv = MultiEnv(setup)

        num_envs = venv.instances
        env_name = next(iter(venv.spec))
        action_space = venv.action_space[env_name].n
        obs_space = self._observation_space(venv, env_name)

        # Check reset
        img = venv.reset()
        self.assertEqual((num_envs, *obs_space), img.shape, msg="Shape mismatch img on reset.")

        # Check step function
        actions = np.random.randint(0, action_space, num_envs)
        img, reward, done, info = venv.step(actions)

        self.assertEqual((num_envs, *obs_space), img.shape, msg="Shape mismatch img.")
        self.assertEqual((num_envs,), reward.shape, msg="Shape mismatch reward.")
        self.assertEqual((num_envs,), done.shape, msg="Shape mismatch done.")
        self.assertEqual((num_envs,), info.shape, msg="Shape mismatch info.")

    @staticmethod
    def _game_loop(setup, nr_games=1, render=False, use_mp=False):
        """ Checks the running of a loop.  """
        venv = MultiEnv(setup, use_multiprocessing=use_mp)
        venv.reset()

        num_envs = venv.instances
        env_name = next(iter(venv.spec))
        action_space = venv.action_space[env_name].n

        episode = 0
        while episode < nr_games:
            actions = np.random.randint(0, action_space, num_envs)
            img, reward, done, info = venv.step(actions)

            for _ in np.where(done)[0]:
                episode += 1

            if render:
                venv.render()

    @staticmethod
    def _observation_space(venv: MultiEnv, env_name: str):
        """ Return the observation space.  """
        environment = venv.env[env_name]

        if isinstance(environment, list):
            # Atari environment return a list and information is stored in shape
            observation_space = environment.pop(0).observation_space.shape

            # 1D environment are converted to a vector.
            if len(observation_space) == 0:
                observation_space = (1,)
        else:
            # procgen stores environment information is spaces
            observation_space = environment.observation_space.spaces['rgb'].shape
        return observation_space
