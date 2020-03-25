"""

Examples showing procgen games.

For convenience the reference `env` will always be used for a single normal
gym environment. The reference `venv` will be used to indicate a vectorized
gym environment.

Whenever we are in a vectorized environment you will need to use vectors and
check the done explicitly, since the basic truth value of an array is ambiguous.

"""

# Required for making a single (non-vector) Procgen environment.
import gym

# The procgen package, creates a vectorized gym environment.
import procgen

# Helper functions to create actions and slow down rendering
import numpy as np
import time


def single_env(env_name: str, fps: int = 60):
    """ Demonstration of making a single procgen environment as if it was a gym env."""

    env = gym.make(f"procgen:procgen-{env_name}-v0")
    env.reset()
    done = False
    score = np.array([0], dtype=np.float32)

    while not done:
        _, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        time.sleep(1 / fps)
        score += reward
    print(f"\nPlayed a game of {env.spec.id}, final score: {next(iter(score))}.")


def single_env_loop(env_name: str, fps: int = 60):
    """ Demonstration of looping over a single procgen environment as if it was a gym env."""

    env = gym.make(f"procgen:procgen-{env_name}-v0")

    nr_games = 5
    print(f"\nGoing to play {nr_games} games of {env.spec.id}")
    for episode in range(nr_games):
        env.reset()
        done = False
        score = np.array([0], dtype=np.float32)

        while not done:
            _, reward, done, _ = env.step(env.action_space.sample())
            env.render()
            time.sleep(1 / fps)
            score += reward
        print(f"\tFinished episode {episode}, final score: {next(iter(score))}.")


def multiple_env(num_envs: int, env_name: str, fps: int = 60):
    """
    Example of how to run ProcGen using VecEnv.

        This environment automatically resets an environment if the game is done.
        So there is no need to call reset when the game is done.
        This is why the loop is run for a certain number of steps.
        In the next example there will be an implementation that counts the number of episodes.
    """

    venv = procgen.ProcgenEnv(num_envs=num_envs, env_name=env_name)
    venv.reset()

    steps = 0
    action_space = venv.action_space.n

    while steps <= 100:
        actions = np.array(np.random.randint(0, action_space, num_envs))
        img, reward, done, info = venv.step(actions)
        venv.render()
        time.sleep(1 / fps)
        steps += 1


def multiple_env_episode_loop(num_envs: int, env_name: str, fps: int = 60):
    """
    Example of how to run ProcGen using VecEnv.

        This environment automatically resets an environment if the game is done.
        So there is no need to call reset when the game is done.
    """

    venv = procgen.ProcgenEnv(num_envs=num_envs, env_name=env_name)
    venv.reset()

    nr_games = 5
    print(f"\nGoing to play {nr_games} games of {venv.options['env_name']}")

    episode = 0
    scores = np.zeros((num_envs,))
    action_space = venv.action_space.n

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
    # For an overview of all the possible procgen games, uncomment next line
    # print("Procgen names:\n\t", "\n\t".join(procgen.env.ENV_NAMES))

    ## Running a single game
    # single_env(env_name='bigfish', fps=60)

    ## Running a loop of games
    # single_env_loop(env_name='bigfish', fps=60)

    ## Running a single game with multiple instances
    # multiple_env(num_envs=12, env_name='bigfish', fps=60)

    ## Running a single game with multiple instances, specified episodes.
    # multiple_env_episode_loop(num_envs=12, env_name='bigfish', fps=60)

    pass
