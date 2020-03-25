from abc import ABC, abstractmethod
from typing import Any, Union


class AbstractPreProcessing(ABC):
    env: Any

    def __init__(self, env, *args, **kwargs):
        pass

    @abstractmethod
    def input_shape(self) -> Union[tuple, list]:
        """ Shape of the model input (after processing).  """
        pass

    @abstractmethod
    def output_shape(self) -> Union[tuple, list, int]:
        """ Shape of the model output.  """
        pass
