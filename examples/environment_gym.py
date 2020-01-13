import numpy as np
import gym

from core.preprocessing.wrappers import GymWrapper
from core.tools import Scheduler

if __name__ == '__main__':
    # This demonstrates how you can wrap a Gym environment like a MultiEnv
    # and use all wrappers for MultiEnv.
    env = gym.make("CartPole-v0")
    env = GymWrapper(env)

    for env, update, episode, step in Scheduler(env, time_limit=5, time_update=3):
        actions = np.array(env.action_space.sample())
        env.step(actions)
