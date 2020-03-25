import warnings
import numpy as np

from core.preprocessing.wrappers import EpisodeStatistics


class Controller:
    """
        This class handles when to print summaries or terminate the Agent.
        This class works together with the EpisodeStatistics class.
    """

    def __init__(self, env,
                 episode_limit=np.uint(-1), step_limit=np.uint(-1), time_limit=np.uint(-1),
                 episode_update=np.uint(-1), step_update=np.uint(-1), time_update=np.uint(-1),
                 write_summary=True, short_summary=True):

        self.env = env
        self.short_summary = short_summary

        if not hasattr(self, 'statistics'):
            warnings.warn("Scheduler requires wrapper `EpisodeStatistics` for summary printing, "
                          "this wrapper is now added to the environment.", UserWarning)
            self.env = EpisodeStatistics(self.env)

        if episode_update >= episode_limit:
            episode_update = episode_limit

        if step_update >= step_limit:
            step_update = step_limit

        if time_update >= time_limit:
            time_update = time_limit

        self.limits = dict(episode=episode_limit, steps=step_limit, time=time_limit)
        self.updates = dict(episode=episode_update, steps=step_update, time=time_update)
        self.update_counts = dict(episode=episode_update, steps=step_update, time=time_update)

        self.reset_images = env.reset()
        self.write_summary = write_summary

        if write_summary:
            self.env.statistics()

    def __getitem__(self, item):
        """ Create an iterator to go over the environment.  """

        env_condition = self.env.summary()
        self._check_loop(env_condition)

        if self.short_summary:
            return self.env
        return self.env, list(env_condition.values())


    def _check_loop(self, env_condition):
        for key, value in self.update_counts.items():
            if env_condition[key] >= value:

                limit = self.limits[key]
                update = self.updates[key]

                # Show statistics and update timer
                self.env.statistics()
                self.update_counts[key] = min(value + update, limit)

                # We reached termination conditions
                if env_condition[key] >= self.limits[key]:
                    raise StopIteration
