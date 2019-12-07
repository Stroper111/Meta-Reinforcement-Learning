
import procgen
import time
import numpy as np

from core import MultiEnv


def multiple_same_env():
    """ Example of how to run ProcGen using VecEnv.  """
    FPS = 60
    N_ENV = 12

    venv = procgen.ProcgenEnv(num_envs=N_ENV, env_name="coinrun")
    venv.reset()

    steps = 0
    action_space = venv.action_space.n
    while steps <= 100:
        img, reward, done, info = venv.step(np.array(np.random.randint(0, action_space, N_ENV)))
        venv.render()
        time.sleep(1 / FPS)
        steps += 1


def multiple_diff_env():
    """ Example of how to use MultiEnv.  """

    # Define dictionary with games and number of instances
    setup = dict(coinrun=2, bigfish=2, chaser=2)

    # Setup the MultiEnv and reset the game for starting
    env = MultiEnv(setup=setup)
    env.reset()
    env.render()

    # predefine all variables
    steps = 0
    img, reward, done, info = [None] * 4
    action_space = env.action_space['coinrun'].n

    # The actual game loop, normally just True
    while steps <= 100:
        # Take some random samples
        actions = np.random.randint(0, action_space, sum(setup.values()))

        # Perform the step
        img, reward, done, info = env.step(actions)

        # Render environment and increase counter
        env.render()
        steps += 1

    # Print the data obtained in the last steps.
    print(img['rgb'].shape)
    print(reward)
    print(done)
    print(info)

    # Slow down for humans
    time.sleep(5)


if __name__ == '__main__':

    # multiple_same_env()
    # multiple_diff_env()

    pass