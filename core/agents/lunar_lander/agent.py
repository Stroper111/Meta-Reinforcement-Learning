import os
import time
import numpy as np
import random as rn

from core.agents import AbstractAgent

from core import MultiEnv
from core.memory import BaseMemoryDeque
from core.memory.sampling import BaseSampling
from .model import LunarLanderModel

# Seed for reproducibility
seed = 1234
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


class LunarLanderDQN(AbstractAgent):
    def __init__(self, setup):
        super().__init__(setup)

        self.env = MultiEnv(setup)
        self.model = LunarLanderModel(input_shape=(8,), output_shape=4)
        self.memory = BaseMemoryDeque(size=1_000_000)
        self.sampler = BaseSampling(self.memory, self.model, gamma=0.99, alpha=0.1, batch_size=64)

        self.eps_min = 0.001

        self.render_interval = 100
        self.checkpoint_interval = 200
        self.save_dir = "checkpoints"

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def run(self):
        msg_interval = 10
        scores = [0 for _ in range(msg_interval)]

        for episode in range(1, 1_000):
            state = self.env.reset()
            done = False

            scores[episode % msg_interval] = 0

            while not done:
                # Epsilon greedy
                action = self.model.actions(state)
                if rn.random() < max(1 / episode, self.eps_min):
                    action = self.model.actions_random(shape=(1,))

                # Add to replay memory
                next_state, reward, done, info = self.env.step(action)

                # Render game
                if episode % self.render_interval == 0:
                    self.env.render()

                reward = reward if not done else -reward
                self.memory.add(state, action, reward, done, next_state)

                # Training step
                if len(self.memory) > self.sampler.batch_size * 2:
                    self.model.train(*self.sampler.random_batch())

                scores[episode % msg_interval] += reward
                state = next_state[:]

            # Print score information.
            scoring = ', '.join(map(lambda score: '% 4d' % score, scores[:(episode % 10)]))
            print("\rEpisode: %4d, score: %s" % (episode, scoring), flush=True, end="")
            if episode % 10 == 0:
                scoring = ', '.join(map(lambda score: '% 4d' % score, scores))
                print("\rEpisode: %4d, score: %s" % (episode, scoring), flush=False)

            # Save checkpoints
            if episode % self.checkpoint_interval == 0:
                self.model.save_checkpoint(self.save_dir, episode=episode)

    @staticmethod
    def _current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')


if __name__ == '__main__':
    setup = {"LunarLander-v2": 1}
    controller = LunarLanderDQN(setup)
    controller.run()
