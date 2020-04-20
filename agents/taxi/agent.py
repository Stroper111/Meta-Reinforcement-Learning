import time
import random

from core.agents import AbstractAgent

from core import MultiEnv
from agents.taxi import TaxiModelEmbedding
from core.memory import BaseMemoryDeque
from core.memory.sampling import BaseSampling


class Taxi(AbstractAgent):
    def __init__(self, setup):
        super().__init__(setup)
        self.instances = sum(setup.values())

        self.env = MultiEnv(setup, use_multiprocessing=True)
        self.model = TaxiModelEmbedding(input_shape=1, output_shape=6)
        self.memory = BaseMemoryDeque(size=500_000)
        # self.memory = BaseMemoryNumpy(size=1_000_000, shape=(1,), action_space=6, stacked_frames=False)
        self.sampler = BaseSampling(self.memory, self.model, gamma=0.99, alpha=0.1, batch_size=200)

    def run(self):
        msg_interval = 10 if self.instances < 10 else self.instances
        scores = [0 for _ in range(msg_interval)]
        scores_temp = [0 for _ in range(self.instances)]

        episode = 0
        episode_temp = 0

        # Resetting is automatically performed by gym wrapper
        state = self.env.reset()

        while episode < 10_000:

            # Epsilon greedy
            action = self.model.act(state)
            if random.random() < max(1 / (episode + 1), 0.05):
                action = self.model.actions_random(shape=(self.instances,))

            # Add to replay memory
            next_state, reward, done, info = self.env.step(action)
            for k in range(self.instances):
                self.memory.add(state[k], action[k], reward[k], done[k], next_state[k])
                scores_temp[k] += reward[k]

                # perform clean up action
                if done[k]:
                    scores[episode % msg_interval] = scores_temp[k]
                    scores_temp[k] = 0
                    episode += 1

            # Training step
            if len(self.memory) > self.sampler.batch_size * 2:
                self.model.train(*self.sampler.random_batch())

            state = next_state[:]

            # Alternative for updating every episode
            if episode_temp != episode:
                episode_temp = episode

                # Print score information.
                if episode % msg_interval == 0:
                    scoring = ', '.join(map(lambda score: '% 5d' % score, scores))
                    print("\rEpisode: %4d, mean %4.3f, score: %s, " % (episode, sum(scores) / len(scores), scoring),
                          flush=False)

    @staticmethod
    def _current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')


if __name__ == '__main__':
    setup = {"Taxi-v3": 5}
    controller = Taxi(setup)
    controller.run()
