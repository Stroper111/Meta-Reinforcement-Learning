import time
import numpy as np

from core.preprocessing import StatisticsUnique


class Scheduler:
    """
        This class handles when to print summaries or terminate the Agent.
        This class works together with the Statistics class.
    """

    def __init__(self, env, episode_limit=np.uint(-1), step_limit=np.uint(-1), time_limit=np.uint(-1),
                 episode_update=np.uint(-1), step_update=np.uint(-1), time_update=np.uint(-1)):

        self.env = env
        self.added_temp = False
        if not hasattr(self.env, 'scheduler'):
            self.added_temp = True
            self.env = StatisticsUnique(self.env, save_dir="temp")

        if episode_update >= episode_limit:
            episode_update = episode_limit

        if step_update >= step_limit:
            step_update = step_limit

        if time_update >= time_limit:
            time_update = time_limit

        self.limits = dict(episode=episode_limit, steps=step_limit, time=time_limit)
        self.updates = dict(episode=episode_update, steps=step_update, time=time_update)
        self.update_counts = dict(episode=episode_update, steps=step_update, time=time_update)

        self.start_time = time.time()
        self.total_time = time.time()

    def __next__(self):

        episodes, steps = self.env.scheduler

        for key, value in [('time',  (time.time() - self.start_time)),
                           ('episode', episodes),
                           ('steps', steps)]:


            if value >= self.updates[key]:
                self.env._write_summary()
                self.updates[key] = min(self.update_counts[key] + self.updates[key], self.limits[key])

                if value >= self.limits[key]:
                    return StopIteration
        return True

    def _write_summary(self):
        self._write(self.env.summary(stats=['mean']))

    def _write(self, msg):
        print(f"\r{msg}", end="")
