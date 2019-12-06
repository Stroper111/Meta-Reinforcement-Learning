
import gym
import time

from procgen.env import ENV_NAMES

print(ENV_NAMES)

for name in ENV_NAMES[:1]:
    env = gym.make(f"procgen:procgen-{name}-v0")
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
