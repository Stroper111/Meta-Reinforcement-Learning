"""
Multiprocessing environment to play games across processes.

The approach is based of this post:
https://squadrick.dev/journal/efficient-multi-gym-environments.html

"""

import gym
import numpy as np

import pickle
import cloudpickle
import multiprocessing as mp

from typing import List
from collections import namedtuple

from .gym_base import BaseGymWrapper


class GymWrapperMP(BaseGymWrapper):
    """ Converter for gym to MultiEnv """
    _step = namedtuple("step", ("img", "reward", "done", "info"))

    def __init__(self, game, instances=1):
        super().__init__()

        # Creating environments
        self._envs = self._create_envs(game, instances)
        self.game = game
        self.instances = instances

        # Remote control values
        self.waiting = False
        self.closed = False

        # instantiate remote workers
        self.conn_parents, self.conn_children = zip(*[mp.Pipe() for _ in range(self.instances)])
        self.processes = self._create_processes(self.conn_children, self.conn_parents, self._envs)

        # Set daemons, in case parent stops, all children get's terminated
        for process in self.processes:
            process.daemon = True
            process.start()

        # Close all connections (this only influence worker if worker is done.)
        for remote in self.conn_children:
            remote.close()

    def __getattr__(self, item):
        return [getattr(each, item) for each in self._envs]

    def _create_envs(self, game, instances) -> List[gym.Env]:
        return [gym.make(game) for _ in range(instances)]

    def _create_processes(self, conn_children, conn_parents, envs):
        processes = []
        for child, parent, env in zip(conn_children, conn_parents, envs):
            process = mp.Process(target=Worker, args=(child, parent, CloudpickleWrapper(env)))
            processes.append(process)
        return processes

    def step_async(self, actions):
        if self.waiting:
            raise ValueError("Already waiting for synchronisation, race condition?")
        self.waiting = True

        # Perform the step asynchronous
        for remote, action in zip(self.conn_parents, actions):
            remote.send(('step', action))

    def step_wait(self):
        if not self.waiting:
            raise ValueError("We are not waiting on a step, race condition?")
        self.waiting = False

        # Receive steps from all cloud processes
        results = [self._step(*remote.recv()) for remote in self.conn_parents]

        # Merge all results (zip is its own inverse)
        results = self._step(*zip(*results))

        # Convert to arrays
        results = self._step(*[np.array(each) for each in results])

        # Reset all environments that are done
        results = self._reset_done(results)

        # Return the results (this is more explicit than results)
        return results.img, results.reward, results.done, results.info

    def step(self, actions: np.array) -> (np.array, np.array, np.array, np.array):
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> np.array:
        for remote in self.conn_parents:
            remote.send(('reset',))
        return np.vstack([remote.recv() for remote in self.conn_parents])

    def render(self):
        for remote in self.conn_parents:
            remote.send(('render',))

    def close(self):
        for remote in self.conn_parents:
            remote.send(('close',))
        self._env = None

    def _reset_done(self, results) -> _step:
        reset_idx = np.where(results.done)[0]
        for idx in reset_idx:
            self.conn_parents[idx].send(('reset',))
            results.img[idx] = self.conn_parents[idx].recv()
        return results


class CloudpickleWrapper(object):
    """ Helper to pass arguments between processes.  """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

    def __call__(self):
        return self.x()


class Worker:
    def __init__(self, remote, parent_remote, env_function):
        parent_remote.close()

        self.env = env_function.x
        self.remote = remote
        done = False

        while not done:
            try:
                command, *data = remote.recv()
                done = getattr(self, command)(*data)
            except EOFError:
                print("EOFError, connection probably terminated?")
                break

        self.remote.close()

    def __getattr__(self, item) -> bool:
        if not hasattr(self.env, item):
            print(f"Worker got request for non-existing attribute `{item}`, terminating")
            return True

        self.remote.send(getattr(self.env, item))
        return False

    def step(self, action):
        self.remote.send(self.env.step(action))
        return False

    def reset(self):
        self.remote.send(self.env.reset(), )
        return False

    def render(self):
        self.env.render()
        return False

    def close(self):
        self.env.close()
        return True


if __name__ == '__main__':
    env = GymWrapperMP("CartPole-v0", 5)
    env.reset()

    for _ in range(50):
        env.step([env.action_space[0].sample() for _ in range(env.instances)])
        env.render()
    env.close()
