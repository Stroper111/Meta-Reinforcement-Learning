import numpy as np

from dataclasses import dataclass

from core.games.pommerman.constants import Action


@dataclass
class Command:
    name: str
    array: np.ndarray
    action: Action


@dataclass
class Commands:
    STOP = Command(name="stop", array=np.array([0, 0]), action=Action.Stop)
    UP = Command(name="up", array=np.array([-1, 0]), action=Action.Up)
    DOWN = Command(name="down", array=np.array([1, 0]), action=Action.Down)
    LEFT = Command(name="left", array=np.array([0, -1]), action=Action.Left)
    RIGHT = Command(name="right", array=np.array([0, 1]), action=Action.Right)
    BOMB = Command(name="bomb", array=np.array([0, 0]), action=Action.Bomb)

    # The output shape (in correct order)
    all = [STOP, UP, DOWN, LEFT, RIGHT, BOMB]

    def get_command(self, name=None, array=None, action=None):
        for identifier, value in [('name', name), ('array', array), ('action', action)]:
            if value is not None and self._get_command(identifier, value) is not None:
                return self._get_command(identifier, value)
        raise ValueError(f"No command found with inputs:\n\tName: {name}\n\tArray: {array}\n\tAction: {action}")

    def _get_command(self, attr, value):
        """ Helper function to check all commands.  """
        for command in self.all:

            # Check normal attributes
            if getattr(command, attr) == value:
                return command

            # Check nested attribute value
            if hasattr(getattr(command, attr), 'value'):
                if getattr(command, attr).value == value:
                    return command

        return None
