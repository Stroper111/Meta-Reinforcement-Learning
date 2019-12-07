import numpy as np
import procgen


class MultiEnv:
    """
        This class controls multiple different game environments.
        It can be used to create an agent that can learn from
        several different environments at the same time.

        setup: dict
            A dictionary where the key is the game name as represented
            in procgen.env.ENV_NAMES and the value is the number of
            instances of this game.
    """

    def __init__(self, setup: dict):
        self.setup = None
        self.instances = None
        self.venv = None

        self.new_setup(setup)

    def __getattr__(self, item):
        """ Back up for invoking an attribute not overruled here, such as action_space.  """
        data = dict()
        for game, venv in zip(self.setup, self.venv):
            if hasattr(venv, item):
                data[game] = getattr(venv, item)
            else:
                raise AttributeError(f"{venv.__class__.__name__} doesn't have attribute '{item}'")
        return data

    @staticmethod
    def create_games(setup):
        """ Helper for creating all environments with the correct number of instances.  """
        venv = []
        for game, instances in setup.items():
            venv.append(procgen.ProcgenEnv(num_envs=instances, env_name=game))
        return venv

    def new_setup(self, setup):
        self.setup = setup
        self.instances = sum(setup.values())
        self.venv = self.create_games(setup)

    def reset(self):
        """ Reset all the environments and return the (stacked) starting images.  """
        images = [venv.reset()['rgb'] for venv in self.venv]
        images = np.vstack(images)
        return dict(rgb=images)

    def step(self, actions):
        """ Performs a step in all environments and combines all results.  """
        action_processed = 0
        actions_per_env = list(self.setup.values())

        # Perform the first steps in the environment, so the remainders can be stacked onto it
        images, rewards, dones, infos = self.venv[0].step(actions[0:actions_per_env[0]])
        images = images['rgb']

        for venv, instances in zip(self.venv[1:], actions_per_env[1:]):
            img, reward, done, info = venv.step(actions[action_processed:action_processed+instances])
            images = np.vstack([images, img['rgb']])
            rewards = np.hstack([rewards, reward])
            dones = np.hstack([dones, done])
            infos.extend(info)
            action_processed += instances

        return dict(rgb=images), rewards, dones, infos

    def render(self):
        """ Render all environments in their own window.  """
        [venv.render() for venv in self.venv]

    def close(self):
        """ Close all the environments.  """
        [venv.close() for venv in self.venv]
        self.venv = None
