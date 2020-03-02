from abc import ABC, abstractmethod
from typing import Union

from core import MultiEnv
from core.preprocessing.wrappers import BaseWrapper


class AbstractAgent(ABC):
    """
        Abstract class to unify all different agents.

        setup: dict
            A dictionary containing as key the game name and as value the
            number of instances of the game.
    """
    env: Union[MultiEnv, BaseWrapper]

    @abstractmethod
    def __init__(self, setup: dict):
        pass

    def __str__(self):
        return '<class {}>'.format(type(self).__name__)

    def __repr__(self):
        return str(self)

    @abstractmethod
    def run(self):
        pass
