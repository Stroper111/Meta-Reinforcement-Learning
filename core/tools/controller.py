import warnings
import numpy as np

from core.preprocessing.wrappers import EpisodeStatistics


# TODO Rebuild
class Scheduler:
    """
        This class handles when to print summaries or terminate the Agent.
        This class works together with the EpisodeStatistics class.
    """

    def __init__(self, env,
                 episode_limit=np.uint(-1), step_limit=np.uint(-1), time_limit=np.uint(-1),
                 episode_update=np.uint(-1), step_update=np.uint(-1), time_update=np.uint(-1),
                 write_summary=True, save_dir=None):

        self.env = env
        self.added_temp = False
        self.save_dir = save_dir

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
        for key, value in self.update_counts.items():

            if env_condition[key] > value:

                # Show statistics and update timer
                self.env.statistics()
                value += self.updates[key]

                # We reached termination conditions
                if env_condition[key] > self.limits[key]:
                    raise StopIteration

        return self.env
