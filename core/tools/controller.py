import time
import numpy as np

from core.preprocessing.wrappers import EpisodeStatistics


# TODO Rebuild
class Scheduler:
    """
        This class handles when to print summaries or terminate the Agent.
        This class works together with the EpisodeStatistics class.
    """

    def __init__(self, env, episode_limit=np.uint(-1), step_limit=np.uint(-1), time_limit=np.uint(-1),
                 episode_update=np.uint(-1), step_update=np.uint(-1), time_update=np.uint(-1),
                 write_summary=True, save_dir=None):

        self.env = env
        self.added_temp = False

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

        self.reset_images = env.reset()
        self.write_summary = write_summary

        if write_summary:
            self._write_summary("Startup", True)

    def __getitem__(self, item):
        episode, steps = self.env.scheduler()
        update = False
        for key, value in [('time', (time.time() - self.start_time)),
                           ('episode', episode),
                           ('steps', steps)]:

            if value >= self.updates[key]:
                if self.write_summary:
                    self._write_summary(key, value)

                self.updates[key] = min(self.update_counts[key] + self.updates[key], self.limits[key])
                update = True
                if value >= self.limits[key]:
                    raise StopIteration
        return self.env, update, episode, steps

    def _write_summary(self, key, value):
        print(f"\nSummary (condition: {key} = {'{:2,d}'.format(int(value))}, elapsed time: {self._convert_time()})")
        print(f"\t{'%-15s' % 'game'}{''.join([('%15s' % key)*instance for key, instance in self.env.setup.items()])}")
        for stat, result in self.env.summary(stats=['mean']).items():
            print(f"\t{stat.ljust(15)}{''.join(['{:15,.2f}'.format(each) for each in result])}")

    def _write(self, msg):
        print(f"\r{msg}", end="")

    def _convert_time(self):
        seconds = int(time.time() - self.start_time)
        minutes = seconds // 60
        hours = minutes // 60
        seconds %= 60
        minutes %= 60
        return '%2d hours %02d minutes %02d seconds' % (hours, minutes, seconds)
