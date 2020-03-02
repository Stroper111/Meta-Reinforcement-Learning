"""
Environment searcher for the gym atari package.

Please make sure that you read the `README.md` at the begin to check
if you have gym[atari] installed. Otherwise this will not work.

"""

from threading import Thread
from queue import Queue

import gym

from collections import namedtuple


def make_env_names():
    names = []
    for each in ['adventure', 'air_raid', 'alien', 'amidar', 'assault',
                 'asterix', 'asteroids', 'atlantis', 'bank_heist', 'battle_zone',
                 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
                 'centipede', 'chopper_command', 'crazy_climber', 'demon_attack',
                 'double_dunk', 'elevator_action', 'enduro', 'fishing_derby', 'freeway',
                 'frostbite', 'gopher', 'gravitar', 'hero', 'ice_hockey', 'jamesbond',
                 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master', 'montezuma_revenge',
                 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan', 'private_eye',
                 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing', 'solaris',
                 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
                 'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
        name = ''.join([g.capitalize() for g in each.split('_')])
        name += '-v0'
        names.append(name)
    return names


class Worker(Thread):
    _result = namedtuple("results", ('name', 'space', 'actions'))

    def __init__(self, load: Queue, results: Queue):
        super().__init__()

        self.load = load
        self.results = results

    def run(self):
        while not self.load.qsize() == 0:
            env_name = self.load.get()
            env = gym.make(env_name)

            name = env_name.ljust(30)
            space = "%-15s" % str(env.reset().shape)
            actions = env.unwrapped.get_action_meanings()

            self.results.put(self._result(name, space, actions))
            env.close()

            print("\rLeft: %3d latest: %s" % (self.load.unfinished_tasks, env_name), flush=True, end="")
            self.load.task_done()


def get_gym_overview():
    env_ids = [env.id for env in gym.envs.registry.env_specs.values()]

    results = Queue()
    load = Queue()

    [load.put(env_id) for env_id in make_env_names()]

    workers = [Worker(load, results) for _ in range(10)]
    [worker.start() for worker in workers]
    [worker.join(timeout=15) for worker in workers]

    print("\rAll results are in, combining them to a new list.")

    result = []
    while not results.qsize() == 0:
        result.append(results.get())

    print("\nSorted on space and actions")

    for name, space, actions in sorted(result, key=lambda x: (len(x[2]), x[2], x[1])):
        print("\t", name, space, actions)


if __name__ == '__main__':
    get_gym_overview()
