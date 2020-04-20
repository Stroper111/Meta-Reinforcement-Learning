import time
import numpy as np

from core.agents import AbstractAgent

from core import MultiEnv
from core.memory import BaseMemoryNumpy, BaseMemoryDeque
from core.memory.sampling import BaseSampling
from core.preprocessing.wrappers import UnpackVec

from agents.cartpole.model import CartPoleModel


class CartPole(AbstractAgent):
    def __init__(self, setup):
        super().__init__(setup)
        self.instances = sum(setup.values())

        self.env = UnpackVec(MultiEnv(setup))
        self.model = CartPoleModel(input_shape=(4,), output_shape=2)
        # self.memory = BaseMemoryDeque(size=1_000_000)
        self.memory = BaseMemoryNumpy(size=1_000_000, shape=(4,), action_space=2, stacked_frames=False)
        self.sampler = BaseSampling(self.memory, self.model, gamma=0.95, alpha=0.1, batch_size=20)

        # Now possible to load model using these examples:
        # Note that custom name has to be the loading directory.
        self.model.create_save_directory(agent_name=self.__class__.__name__, game_name='cartpole', custom_name="")
        # self.model.load_checkpoint(load_name='last')

    def run(self):
        msg_interval = 10 if self.instances < 10 else self.instances
        scores = [0 for _ in range(msg_interval)]

        for episode in range(1, 5001):
            done = False
            state = self.env.reset()
            scores_temp = 0

            while not done:
                state = np.expand_dims(state, axis=0)
                action = self.model.act(state)

                next_state, reward, done, info = self.env.step(action)
                self.memory.add(state, action, reward, done, next_state)
                scores_temp += reward

                # Training step
                if len(self.memory) > self.sampler.batch_size * 2:
                    self.model.train(*self.sampler.random_batch())

                state = next_state[:]

            scores[(episode - 1) % msg_interval] = scores_temp

            # Print score information.
            scoring = ', '.join(map(lambda score: '%4d' % score, scores[:(episode % msg_interval)]))
            print("\rEpisode: %4d, score: %s" % (episode, scoring), flush=True, end="")

            if episode % msg_interval == 0:
                scoring = ', '.join(map(lambda score: '%4d' % score, scores))
                # self.model.save_checkpoint(model=self.model, save_name="episode %4d" % episode)
                print("\rEpisode: %4d, score: %s" % (episode, scoring), flush=False)

        # self.model.save_model(save_name="episode %4d" % episode)


if __name__ == '__main__':
    setup = {"CartPole-v0": 1}
    controller = CartPole(setup)
    controller.run()
