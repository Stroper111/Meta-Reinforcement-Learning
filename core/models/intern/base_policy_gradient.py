"""
Base class for policy gradient, based on the github repository:
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

From karpathy story:
https://karpathy.github.io/2016/05/31/rl/


It provides a way to create different layers and apply gradients to them.

"""
import numpy as np
import os
import pickle
import warnings

from typing import Union, List

from core.models import BaseModel


class BaseModelPG(BaseModel):
    checkpoint: str = "checkpoints"
    dir_model: str = "pretrained"
    dir_load: str = None
    dir_save: str = None

    model: List['Layer']

    __slots__ = ('model')

    def __init__(self, input_shape, output_shape,
                 hidden_layers: list = None, initialization='xavier', *args, **kwargs):
        super().__init__(input_shape, output_shape)

        # Determine input and output shape of the model
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Determine the number of neurons per hidden layer (every entry is a layer)
        self.hidden_layers = hidden_layers
        self.initialization = initialization

        # Create the model
        self.model: List['Layer'] = self.create_model(input_shape, output_shape,
                                                      hidden_layers, initialization,
                                                      *args, **kwargs)

    @staticmethod
    def create_model(input_shape, output_shape, *args, **kwargs):
        """ Creates the model, by adding layers together.  """
        (first_layer, *hidden_layers), initialization, *_ = args

        layers = []
        layers.append(Layer("W0", input_shape=(first_layer, *input_shape), initialization=initialization))

        for idx, hidden_layer in enumerate(hidden_layers, start=1):
            layers.append(Layer(name="W%d" % idx, input_shape=(hidden_layer,), initialization=initialization))

        return layers

    @property
    def layer_names(self):
        return [layer.name for layer in self.model]

    @property
    def layer_weights(self):
        return [layer.weights for layer in self.model]

    @property
    def layers(self):
        return {name: weight for name, weight in zip(self.layer_names, self.layer_weights)}

    def predict(self, states):
        """ Return the predictions of your model.  """
        raise NotImplemented

    def actions(self, states):
        """ Return the actions to execute, this can be combined with epsilon.  """
        raise NotImplemented

    def actions_random(self, shape: tuple):
        return np.random.randint(0, self.output_shape, shape)

    def train(self, x, y):
        """ Trains a model. """
        raise NotImplemented

    def _save_model(self, save_path: str, *args, **kwargs):
        """ Saves the actual model.  """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path, "wb") as file:
            pickle.dump(self.model, file)

    def _load_model(self, load_path: str, *args, **kwargs):
        """ Loads the actual model.  """
        if not os.path.exists(load_path):
            print(f"Unable to load model, path does not exist...\n\t{load_path}\n")
        else:
            with open(load_path, "rb") as file:
                self.model = pickle.load(file)
            print(f"Successfully loaded model...\n\t{load_path}\n")
        return load_path


class Layer:
    def __init__(self, name: str, input_shape: Union[tuple, list], initialization: str = 'randn'):
        self.name = name
        self.input_shape = input_shape
        self.initialization = initialization
        self._weights = self._init_weights(self.input_shape, self.initialization)

        self.gradients = np.zeros_like(self._weights)
        self.rmsprop = np.zeros_like(self._weights)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray):
        self._weights = weights

    def _init_weights(self, input_shape: Union[tuple, list], initialization: str):

        valid_initializations = []
        for each in dir(self):
            if each.startswith("_init_"):
                valid_initializations.append(each[6:])

        if initialization not in valid_initializations:
            valid = '\n\t'.join(valid_initializations)
            raise ValueError(f"Initialization is unknown, valid initializations are:\n\t{valid}")
        return getattr(self, "_init_" + initialization)(input_shape)

    def apply_gradient(self, decay_rate: float, learning_rate: float):
        """ perform rmsprop parameter update.  """
        self.rmsprop = decay_rate * self.rmsprop + (1 - decay_rate) * self.gradients ** 2
        self._weights += learning_rate * self.gradients / (np.sqrt(self.rmsprop) + 1e-5)
        self.gradients.fill(0)

    def _init_zeros(self, input_shape):
        return np.zeros(input_shape)

    def _init_ones(self, input_shape):
        return np.ones(input_shape)

    def _init_randn(self, input_shape):
        return np.random.randn(*input_shape)

    def _init_xavier(self, input_shape):
        if len(input_shape) > 2:
            warnings.warn("Xavier initialization will take last dimension for scaling!", UserWarning)
        return np.random.randn(*input_shape) / np.sqrt(input_shape[-1])
