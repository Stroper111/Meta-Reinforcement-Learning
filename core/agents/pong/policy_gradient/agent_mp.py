"""

Reproduction of the following post, but transferred to our framework.


Numpy implementation:
http://karpathy.github.io/2016/05/31/rl/

Tensorflow implementation:
https://github.com/gameofdimension/policy-gradient-pong



"""
import numpy as np
import random
import os
import re

from typing import List
from core.agents import AbstractAgent

from core import MultiEnv
from core.agents.pong.policy_gradient.preprocessing import PongWrapper
from core.agents.pong.policy_gradient.model import PongModelPG

# Seed for reproducibility (this is not guaranteed using Keras/PyTorch)
seed = 1234
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


class PongPG(AbstractAgent):
    input_shape = (80 * 80,)
    output_shape = 2
    hidden_layers = [200, 200]
    initialization = 'xavier'

    learning_rate = 1e-3
    gamma = 0.99
    decay_rate = 0.99
    batch_size = 10

    render = False
    render_interval = 5

    resume = True
    save = True
    save_interval = 100
    save_name = 'episode {:6d}, mean {: 8.5f}.pkl'

    def __init__(self, setup):
        super().__init__(setup)

        self.env = MultiEnv(setup, use_multiprocessing=True)
        self.env = PongWrapper(self.env)

        self.model = PongModelPG(self.input_shape, self.output_shape, self.hidden_layers, self.initialization)
        self.model.create_save_directory(agent_name=self.__class__.__name__,
                                         game_name='Pong-v0',
                                         custom_name='baseline')

        if self.resume:
            self.save_path = self.model.load_checkpoint('last')

    @staticmethod
    def _locate_int_offset(string: str, search_pattern: str, match_idx: int, alternative: int):
        """ Locate offsets inside the save name, if none is found it will be zero (no offsets).  """
        matches = re.findall(search_pattern, string)
        return int(matches[match_idx]) if len(matches) > match_idx else alternative

    def run(self):

        episode = 0
        running_reward = -21  # Specific for pong

        # Calculate offsets to the episode and running reward based on save game.
        if hasattr(self, 'save_path'):
            save_name = os.path.basename(self.save_path)
            episode = self._locate_int_offset(save_name, '\d+', match_idx=0, alternative=0)
            running_reward = self._locate_int_offset(save_name, '\d+', match_idx=1, alternative=-21)

        obs = self.env.reset()
        obs_prev = None

        reward_sum = np.zeros((self.env.instances,))
        actions = np.zeros((self.env.instances,), dtype=np.uint8)

        # This is our memory that we will use later on.
        obs_: List[List[np.ndarray]] = [[] for _ in range(self.env.instances)]
        hidden_: List[List[np.ndarray]] = [[] for _ in range(self.env.instances)]
        dlogps_: List[List[np.ndarray]] = [[] for _ in range(self.env.instances)]
        discounted_: List[List[np.ndarray]] = [[] for _ in range(self.env.instances)]

        while True:
            if self.render and episode % self.render_interval == 0:
                self.env.render()

            # Take the difference between observations as input
            X = obs - obs_prev if obs_prev is not None else np.zeros_like(obs)
            obs_prev = obs

            # Get action for every game and store intermediate values
            for idx, x_ in enumerate(X):
                action_probability, hidden = self.model.policy_forward(x_)
                actions[idx] = 2 if np.random.uniform() < action_probability else 3

                # Record various intermediates, that are needed for the back propagation.
                obs_[idx].append(x_)
                hidden_[idx].append(hidden)

                # We create a fake label (1, 0) depending on the action we choose
                # Than we encourage the action that was taken to be taken.
                # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
                dlogps_[idx].append(int(actions[idx] == 2) - action_probability)

            # Note that we are still in a multi environment and we need to return a list/array
            obs_next, reward, done, info = self.env.step(actions)
            obs = obs_next

            # Record reward (has to be done after we call step() to get reward for previous action)
            [discounted_[idx].append(each) for idx, each in enumerate(reward)]

            # Bookkeeping
            reward_sum += reward

            # # Pong specific, a point is scored
            # if reward:
            #     print(f"\r\tEpisode %6d, reward: % 3d" % (episode, reward_sum), flush=True, end='')

            for idx in np.where(done)[0]:
                # We finished a run, and now need to calculate all kind of stuff ...
                episode += 1
                reward[idx] = 0

                running_reward = 0.99 * running_reward + 0.01 * reward_sum[idx]

                # Get all values
                episode_obs = np.vstack(obs_[idx])
                episode_hidden = np.vstack(hidden_[idx])
                episode_logps = np.vstack(dlogps_[idx])
                episode_rewards = np.vstack(discounted_[idx])

                # compute the discounted reward backwards through time
                discounted_episode_reward = self.model.discount_rewards(episode_rewards, gamma=self.gamma)

                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_episode_reward -= np.mean(discounted_episode_reward)
                discounted_episode_reward /= np.std(discounted_episode_reward)

                # modulate the gradient with advantage (policy gradient magic happens right here.)
                episode_logps *= discounted_episode_reward

                # Calculate all the backward gradients and store them in the layers
                self.model.policy_backward(episode_hidden, episode_logps, episode_obs)

                if episode % self.batch_size == 0:
                    self.model.apply_gradients(decay_rate=self.decay_rate, learning_rate=self.learning_rate)

                if self.save and episode % self.save_interval == 0:
                    self.model.save_checkpoint(save_name=self.save_name.format(episode, running_reward))

                # Update information for the user.
                msg = '\rEpisode {:6d}, reward: {: 3d} running mean: {: 8.5f}'
                print(msg.format(episode, int(reward_sum[idx]), running_reward))

                # Reset all memories
                obs_[idx] = []
                hidden_[idx] = []
                dlogps_[idx] = []
                discounted_[idx] = []
                reward_sum[idx] = 0


if __name__ == '__main__':
    setup_ = {"Pong-v0": 8}
    controller = PongPG(setup_)
    controller.run()
