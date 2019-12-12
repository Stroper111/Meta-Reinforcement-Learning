import numpy as np
import keras

from abc import ABC, abstractmethod
from typing import Union

from keras import Sequential, Input
from core.memory import BaseSampling


class AbstractModel(ABC):
    """
        Implements a Neural Network using keras

        input_shape:
            The input shape for the model
        action_space:
            The output shape of the model
        setup:
            A dictionary containing all setup information
            layers: layer name = layer settings
            optimizer: name = optimizer settings
            compile: name = value (except the optimizer)
    """

    def __init__(self, input_shape: Union[tuple, list], action_space: int):
        pass

    @staticmethod
    @abstractmethod
    def create_model(input_shape, action_space) -> keras.models:
        """ Creates the model.  """
        pass

    @abstractmethod
    def predict(self, states: np.array) -> np.array:
        """ Returns the prediction of a model.  """
        pass

    @abstractmethod
    def train(self, sampling: BaseSampling):
        """ Trains a model on the sampling.  """
        pass

    @abstractmethod
    def save_model(self, save_dir: str, episode, steps):
        """ Store the model to a file.  """
        pass

    @abstractmethod
    def save_checkpoint(self, save_dir: str, episode, steps):
        """ Store the model and metadata to continue a training session.  """
        pass

    @abstractmethod
    def load_model(self, load_dir: str):
        """ Load a keras model.   """
        pass

    @abstractmethod
    def load_checkpoint(self, load_dir: str):
        """ Continues a training run from the checkpoint.  """
        pass
