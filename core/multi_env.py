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
        self.setup = setup
        self.instances = sum(setup.values())
        self.venv = self._create_games(setup)

    def _create_games(self, setup):
        """ Helper for creating all environments with the correct number of instances.  """
        venv = []
        for game, instances in setup.items():
            venv.append(procgen.ProcgenEnv(num_envs=instances, env_name=game))
        return venv

    def reset(self):
        """ Reset all the environments and return the (stacked) starting images.  """
        images = [venv.reset()['rgb'] for venv in self.venv]
        images = np.vstack(images)
        return dict(rgb=images)

    def step(self, actions):
        """ Performs a step in all environments and combines all results.  """
        action_processed = 0
        actions_per_env = list(self.setup.values())

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
