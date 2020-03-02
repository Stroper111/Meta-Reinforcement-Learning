"""
This is an example of how a basic gym game loop looks like for one game.
And for running multiple episodes.
"""

import gym

if __name__ == '__main__':
    env = gym.make(id="MsPacman-v0")

    # A single loop example

    done = False
    env.reset()
    score = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(actions=action)
        env.render()
        score += reward
    env.close()

    print(f"\nFinished one game of {env.spec.id}, with a score of {score}.")

    # A multi loop example
    nr_games = 5
    print(f"\nPlaying {nr_games} games of {env.spec.id}:")
    for episode in range(1, nr_games + 1):
        done = False
        env.reset()
        score = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(actions=action)
            score += reward

        print(f"\tFinished episode: {episode}, score: {'%4d' % score}")
