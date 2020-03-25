"""

Make sure that you understand how the loops work in gym and procgen.

The multi environment combines gym with procgen and also adds the option to
include multiple different games for Meta Reinforcement Learning, this is
demonstrated in the file `multi_env_extra`.

Here is demonstrate how to use MultiEnv to create gym and procgen games.

"""

import numpy as np
import time

import os, sys

## Adding core directory, for terminal use
directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directory)

from core import MultiEnv


def timer(func):
    """ Wrapper to time performance.  """

    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        returns = func(*args, **kwargs)
        print(f"\nTime taken to run {func.__name__}: {'%2.5f' % (time.time() - start_time)} sec.")
        print(f"\tArgs: {args}", f"\n\tKwargs: {kwargs}\n")
        return returns

    return timer_wrapper


@timer
def multi_environment(setup: dict, fps=60):
    """
        This runs a single gym game with the option to select multiple instances.
    """

    # Note that venv is a vectorized gym environment.
    venv = MultiEnv(setup)

    num_envs = venv.instances
    env_name = next(iter(venv.spec))

    venv.reset()

    episode = 0
    scores = np.zeros((num_envs,))
    action_space = venv.action_space[env_name].n

    nr_games = 5
    # Since it is possible to get multiple games the env specs are returned as dict[str, list]
    print(f"\nPlaying {num_envs} instance(s) of {env_name}, for {nr_games} games.")

    while episode < nr_games:
        actions = np.array(np.random.randint(0, action_space, num_envs))
        img, reward, done, info = venv.step(actions)

        scores += reward

        for idx in np.where(done)[0]:
            print(f"\tFinished episode {episode}, final score: {scores[idx]}")
            episode += 1
            scores[idx] = 0

        venv.render()
        time.sleep(1 / fps)


if __name__ == '__main__':
    ## Running a single gym game with MultiEnv
    # setup = {"CartPole-v0": 1}
    # multi_environment(setup, fps=60)

    ## Running multiple gym instances with MultiEnv
    # setup = {"CartPole-v0": 5}
    # multi_environment(setup, fps=60)

    ## Running a single procgen game with MultiEnv
    # setup = dict(bigfish=1)
    # multi_environment(setup, fps=60)

    ## Runningmultiple procgen instances with Multi env
    # setup = dict(bigfish=5)
    # multi_environment(setup, fps=60)

    pass
