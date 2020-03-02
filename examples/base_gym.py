"""
Basic gym game loop implementation
"""

import time

# Get access to the gym package
import gym

if __name__ == '__main__':
    # For an overview of all possible games uncomment the next line
    # print("All gym games:\n\t", "\n\t".join([each for each in gym.envs.registry.env_specs.keys()]))

    env = gym.make(id="CartPole-v0")

    # We always have to reset a game to start
    env.reset()

    # We can get a random action using
    action = env.action_space.sample()

    # Executing an action returns a tuple of information
    obs, reward, done, info = env.step(actions=action)

    # To render an environment  (can be text or image, depending on environment)
    env.render()

    # Slow down so we can actually see what happens
    time.sleep(3)

    # To terminate use
    env.close()
