"""

Reproduction of the following post, but transferred to our framework.


Numpy implementation:
http://karpathy.github.io/2016/05/31/rl/

Tensorflow implementation:
https://github.com/gameofdimension/policy-gradient-pong



"""

import os
import random
import numpy as np
import re

from core.agents import AbstractAgent
from core.preprocessing.wrappers import UnpackVec

from core import MultiEnv
from core.agents.pong.policy_gradient import PongWrapper, PongModelPG

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
    save_interval = 1
    save_name = 'episode {:6d}, mean {: 8.5f}.pkl'

    def __init__(self, setup):
        super().__init__(setup)

        self.env = MultiEnv(setup)
        self.env = PongWrapper(self.env)
        self.env = UnpackVec(self.env)  # Remove this to work with multiple environments later

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

        while True:
            obs = self.env.reset()
            obs_prev = None

            done = False
            episode += 1
            reward_sum = 0
            reward_msg = ''

            # This is our memory that we will use later on.
            obs_, hidden_, dlogps_, discounted_ = [], [], [], []

            while not done:
                if self.render and episode % self.render_interval == 0:
                    self.env.render()

                # Take the difference between observations as input
                X = obs - obs_prev if obs_prev is not None else np.zeros_like(obs)
                obs_prev = obs

                action_probability, hidden = self.model.policy_forward(X)
                action = 2 if np.random.uniform() < action_probability else 3

                # Record various intermediates, that are needed for the back propagation.
                obs_.append(X)
                hidden_.append(hidden)

                # We create a fake label (1, 0) depending on the action we choose
                # Than we encourage the action that was taken to be taken.
                # (see http://cs231n.github.io/neural-networks-2/#losses if confused)
                dlogps_.append(int(action == 2) - action_probability)

                # Note that we are still in a multi environment and we need to return a list/array
                obs, reward, done, info = self.env.step([action])

                # Record reward (has to be done after we call step() to get reward for previous action)
                discounted_.append(reward)

                # Bookkeeping
                reward_sum += reward

                # Pong specific, a point is scored
                if reward:
                    reward_msg += '+' if reward > 0 else '-'
                    msg = "\r\tEpisode {:6d}, reward: {: 3d}, points: {:42s}"
                    print(msg.format(episode, int(reward_sum), reward_msg), flush=True, end='')

            # We finished a run, and now need to calculate all kind of stuff ...
            running_reward = 0.99 * running_reward + 0.01 * reward_sum

            episode_obs = np.vstack(obs_)
            episode_hidden = np.vstack(hidden_)
            episode_logps = np.vstack(dlogps_)
            episode_rewards = np.vstack(discounted_)

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
            print(msg.format(episode, int(reward_sum), running_reward))


if __name__ == '__main__':
    setup_ = {"Pong-v0": 1}
    controller = PongPG(setup_)
    controller.run()
