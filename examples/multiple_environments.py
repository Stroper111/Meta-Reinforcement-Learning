
import procgen
import time
import numpy as np

from core import MultiEnv


# Running multiple instances of the same environment
def multiple_same_env():
    FPS = 10
    N_ENV = 12

    venv = procgen.ProcgenEnv(num_envs=N_ENV, env_name="coinrun")
    venv.reset()
    action_space = venv.action_space.n

    while True:
        img, reward, done, info = venv.step(np.array(np.random.randint(0, action_space, N_ENV)))
        venv.render()
        time.sleep(1 / FPS)


# Running multiple instances of different environment
def multiple_diff_env():
    setup = dict(coinrun=2, bigfish=2, chaser=2)
    env = MultiEnv(setup=setup)
    env.reset()
    env.render()

    img, reward, done, info = env.step(np.array([0] * 4))

    print(img['rgb'].shape)
    print(reward)
    print(done)
    print(info)

    env.render()
    time.sleep(5)


multiple_diff_env()
