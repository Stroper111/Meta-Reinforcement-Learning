
import gym
import time

from procgen.env import ENV_NAMES


if __name__ == '__main__':
    FPS = 60

    for name in ENV_NAMES[:2]:
        env = gym.make(f"procgen:procgen-{name}-v0")
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
            env.render()
            time.sleep(1/FPS)
