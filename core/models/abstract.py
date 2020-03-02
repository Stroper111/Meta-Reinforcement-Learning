import numpy as np
import keras

from abc import ABC, abstractmethod
from typing import Union


class AbstractModel(ABC):
    """
        Implements a Neural Network using keras

        input_shape:
            The input shape for the model
        output_shape:
            The output shape of the model
        setup:
            A dictionary containing all setup information
            layers: layer name = layer settings
            optimizer: name = optimizer settings
            compile: name = value (except the optimizer)
    """
    model: keras.models

    def __init__(self, input_shape: Union[tuple, list, int], output_shape: Union[tuple, list, int]):
        pass

    @staticmethod
    @abstractmethod
    def create_model(input_shape: Union[tuple, list], output_shape: Union[tuple, list, int]) -> keras.models:
        """ Creates the model.  """
        pass

    @abstractmethod
    def predict(self, states) -> np.array:
        """ Return the predictions of your model.  """
        pass

    @abstractmethod
    def actions(self, states) -> np.array:
        """ Return the actions to execute, this can be combined with epsilon.  """
        pass

    @abstractmethod
    def train(self, x: np.array, y: np.array):
        """ Trains a model on the sampling.  """
        pass

    @abstractmethod
    def save_model(self, save_name: str, weights_only: bool):
        """ Store the model to a file.  """
        pass

    @abstractmethod
    def save_checkpoint(self, save_name: str, weights_only: bool):
        """ Store the model and metadata to continue a training session.  """
        pass

    @abstractmethod
    def load_model(self, load_name: str, weights_only: bool):
        """ Load a keras model.   """
        pass

    @abstractmethod
    def load_checkpoint(self, load_name: str, weights_only: bool):
        """ Continues a training run from the checkpoint.  """
        pass
