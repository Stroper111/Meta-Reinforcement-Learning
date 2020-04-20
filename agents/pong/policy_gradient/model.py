import numpy as np
import warnings
import pickle
import os

from typing import List

from core.models import BaseModelPG
from core.models.intern import Layer


class PongModelPG(BaseModelPG):

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
        probability = self.sigmoid(log_probability)

        # return probability of taking action 2, and hidden state
        return probability, hidden

    def policy_backward(self, hidden: np.ndarray, action_loss: np.ndarray, episode_obs: np.ndarray):
        """ Applying back propagation. Note that the gradients are stored inside the respectivee layer.   """

        grad_weight1 = np.dot(hidden.T, action_loss).ravel()
        self.model[1].gradients += grad_weight1

        d_hidden = np.outer(action_loss, self.model[1].weights)
        d_hidden = self.relu(d_hidden)  # backpro prelu

        grad_weight0 = np.dot(d_hidden.T, episode_obs)
        self.model[0].gradients += grad_weight0

    def _load_model(self, load_path: str, *args, **kwargs):
        """ Loads the actual model.  """
        if not os.path.exists(load_path):
            print(f"Unable to load model, path does not exist...\n\t{load_path}\n")
        else:
            with open(load_path, "rb") as file:
                new_model = pickle.load(file)

            # If importing from original
            if isinstance(new_model, dict):
                self.model: List['Layer'] = self._load_model_dict(new_model)
            elif isinstance(new_model, list):
                self.model = new_model
            else:
                raise ValueError(f"Type of new model not recognized: `{type(new_model)}`")

            print(f"Successfully loaded model...\n\t{load_path}\n")
        return load_path

    def _load_model_dict(self, model):
        """ This is a conversion from original to new implementation.  """

        if len(model) != len(self.layer_names):
            warnings.warn("Loaded model is not the same as created model!", UserWarning)

        new_model = []
        for name, weight in model.items():
            layer = Layer(name, weight.shape)
            layer.weights = weight
            new_model.append(layer)
        return new_model
