import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Union


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
    model: Any

    def __init__(self,
                 input_shape: Union[tuple, list, int],
                 output_shape: Union[tuple, list, int],
                 *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def create_model(input_shape: Union[tuple, list],
                     output_shape: Union[tuple, list, int],
                     *args, **kwargs
                     ) -> Any:
        """ Creates the model.  """
        pass

    @property
    def W1(self):
        return None

    @abstractmethod
    def predict(self, states) -> np.array:
        """ Return the predictions of your model.  """
        pass

    @abstractmethod
    def act(self, states) -> np.array:
        """ Return the actions to execute, this can be combined with epsilon.  """
        pass

    @abstractmethod
    def train(self, x: np.array, y: np.array):
        """ Trains a model on the sampling.  """
        pass

    @abstractmethod
    def save_checkpoint(self, save_name: str, *args, **kwargs):
        """ Store the model and metadata to continue a training session.  """
        pass

    @abstractmethod
    def save_model(self, save_name: str, *args, **kwargs):
        """ Store the model to a file.  """
        pass

    @abstractmethod
    def load_checkpoint(self, load_name: str, *args, **kwargs) -> str:
        """ Continues a training run from the checkpoint.  """
        pass

    @abstractmethod
    def load_model(self, load_name: str, *args, **kwargs) -> str:
        """ Load a keras model.   """
        pass
