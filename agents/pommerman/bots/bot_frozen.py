from core.games.pommerman import characters
from core.games.pommerman.constants import Action
from core.games.pommerman.agents import BaseAgent


class BotFreeze(BaseAgent):
    """ Our version of the base agent. """

    def __init__(self, character=characters.Bomber):
        super().__init__(character)

    def act(self, obs, action_space):
        # Main event that is being called on every turn.
        return Action.Stop
