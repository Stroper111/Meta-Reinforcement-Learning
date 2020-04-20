import os
import pickle

import numpy as np

from core.models import BaseModelPG


class PommermanPGModel(BaseModelPG):

    def apply_gradients(self, decay_rate: float, learning_rate: float):
        [layer.apply_gradient(decay_rate=decay_rate, learning_rate=learning_rate) for layer in self.model]

    @staticmethod
    def discount_rewards(rewards: np.ndarray, gamma: float):
        """ Incrementing reward over time.  """
        discounted = np.zeros_like(rewards)
        running_add = 0

        # We are looping over it backwards, so we can propagate the rewards down.
        for idx, reward in enumerate(reversed(rewards), start=1):
            if reward != 0:
                running_add = 0  # Reset the game, after every point (pong specific!)

            running_add = running_add * gamma + reward
            discounted[len(discounted) - idx] = running_add

        return discounted

    def policy_forward(self, obs):
        hidden = np.dot(self.model[0].weights, obs)
        hidden = self.relu(hidden)
        log_probability = np.dot(self.model[1].weights, hidden)
        probabilities = self.softmax(log_probability)
        return probabilities, hidden

    def policy_backward(self, hidden: np.ndarray, action_loss: np.ndarray, episode_obs: np.ndarray):
        """ Applying back propagation. Note that the gradients are stored inside the respective layer.   """

        grad_weight1 = np.dot(hidden.T, action_loss)
        self.model[1].gradients += grad_weight1.transpose([1, 0])

        d_hidden = np.outer(action_loss, self.model[1].weights)
        d_hidden = self.derivative_relu(d_hidden)  # backpro prelu

        grad_weight0 = np.dot(d_hidden.T, np.repeat(episode_obs, 6, axis=0))
        x, y = grad_weight0.shape
        grad_weight0 = np.sum(grad_weight0.reshape((6, x // 6, y)))
        self.model[0].gradients += grad_weight0

    def _load_model(self, load_path: str, *args, **kwargs):
        if not os.path.exists(load_path):
            print(f"Unable to load model, path does not exist...\n\t{load_path}\n")
        else:
            with open(load_path, "rb") as file:
                self.model = pickle.load(file)
        return load_path
