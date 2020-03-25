"""
In this file are examples that are unique to the MultiEnv.

First we are going to show how we can create multiple different games.
And after that how we can use multiprocessing to start using multiple cores
on the computer.
"""

import time
import numpy as np

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
def multi_environment(setup: dict, fps: int = 60, use_mp: bool = False):
    num_envs = sum(setup.values())

    venv = MultiEnv(setup, use_multiprocessing=use_mp)
    venv.reset()

    episode = 0
    scores = np.zeros((num_envs,))
    action_space = venv.action_space[next(iter(setup))].n

    nr_games = 5

    games = []
    for game, instances in setup.items():
        print(f"Playing {instances} instance(s) of {game}, for at most {nr_games} games.")
        for _ in range(instances):
            games.append(game)

    while episode < nr_games:
        actions = np.array(np.random.randint(0, action_space, num_envs))
        img, reward, done, info = venv.step(actions)

        scores += reward

        for idx in np.where(done)[0]:
            print(f"\tFinished episode {episode}, game: {'%-17s' % games[idx]}, final score: {scores[idx]}")
            episode += 1
            scores[idx] = 0

        venv.render()
        time.sleep(1 / fps)
    venv.close()


def mixed():
    raise NotImplementedError("Not available, due to differrent game size, "
                              "if obs and actions are the same feel free to try.")


if __name__ == '__main__':
    ## Running different gym games at the same time (please note that actions and input space should be the same)
    ## For examples of same data take a look at `extra/environment info.txt`
    # multi_environment({'Pong-v0': 1, 'SpaceInvaders-v0': 1}, fps=999)

    ## Running different procgen games at the same time
    ## Due to the way procgen is build, all environments are compatible with each other.
    # multi_environment(dict(coinrun=1, bigfish=1), fps=999)

    ## Whenever you want to play many games at the same time in a gym, running it on one thread is not always good.
    ## By using the use multiprocessing statement in the MultiEnv all gym games run in a separate process.
    ## Please note that this is already the case in the procgen environment.
    ## The only difference is in the background, for high amount of games mp might be better.
    ## In case of doubt, don't use it.
    # multi_environment({'Pong-v0': 5}, fps=999, use_mp=True)

    ## This can also be used for different gym games (if actions and observation space are similar)
    # multi_environment({'Pong-v0': 2, 'SpaceInvaders-v0': 2}, fps=999, use_mp=True)

    pass
