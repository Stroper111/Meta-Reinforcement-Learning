from abc import ABC


class AbstractAgent(ABC):
    """
        Abstract class to unify all different agents.

        setup: dict
            A dictionary containing as key the game name and as value the
            number of instances of the game.
    """

    def __init__(self, setup: dict):
        pass

    def __str__(self):
        return '<class {}>'.format(type(self).__name__)

    def __repr__(self):
        return str(self)

    def run(self):
        pass
