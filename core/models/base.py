import numpy as np
import pathlib
import time
import os
import glob

from typing import Any
from core.models import AbstractModel


class BaseModel(AbstractModel):
    checkpoint: str = "checkpoints"
    dir_model: str = "pretrained"
    dir_load: str = None
    dir_save: str = None

    model: Any

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super().__init__(input_shape, output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.model = None

    @staticmethod
    def create_model(input_shape, output_shape, *args, **kwargs):
        """ Creates the model.  """
        raise NotImplementedError

    def predict(self, states):
        """ Return the predictions of your model.  """
        raise NotImplementedError

    def actions(self, states):
        """ Return the actions to execute, this can be combined with epsilon.  """
        raise NotImplementedError

    def actions_random(self, shape: tuple):
        return np.random.randint(0, self.output_shape, shape)

    def train(self, x, y):
        """ Trains a model. """
        raise NotImplementedError

    def create_save_directory(self, agent_name: str = None, game_name: str = None, custom_name: str = None):
        """
        Creates the save directory and checkpoint path based on the given inputs

        :param agent_name: Name of the Agent, top directory
        :param game_name: Name of the game being played, subdirectory
        :param custom_name: Save directory, if None the current time will be used.
        """

        # Set all variables
        dir_agent = "base" if agent_name is None else agent_name
        dir_game = "unknown" if game_name is None else game_name
        custom_name = self._current_time() if custom_name is None else custom_name

        # Get to top directory
        dir_package = str(pathlib.Path(os.path.abspath(__file__)).parents[2])

        # Set save and checkpoint directory
        self.dir_save = os.path.join(dir_package, self.dir_model, dir_agent, dir_game, custom_name)
        self.dir_load = self.dir_save

    def save_model(self, save_name: str, *args, **kwargs):
        """ This stores the model with given save name at the created directory.  """
        assert self.dir_save, "No saving directory found, initialize it before loading or run `create_save_directory`"
        save_path = os.path.join(self.dir_save, save_name)
        self._save_model(save_path)

    def load_model(self, load_name: str = 'last', *args, **kwargs) -> str:
        assert self.dir_load, "No loading directory found, initialize it before loading or run `create_save_directory`"
        load_path = self._get_files(dir_load=self.dir_load, load_name=load_name)
        return self._load_model(load_path)

    def save_checkpoint(self, save_name: str, *args, **kwargs):
        """ This stores the model with given save name at the created directory.  """
        assert self.dir_save, "No saving directory found, initialize it before loading or run `create_save_directory`"
        save_path = os.path.join(self.dir_save, self.checkpoint, save_name)
        self._save_model(save_path)

    def load_checkpoint(self, load_name: str = 'last', *args, **kwargs) -> str:
        """ This loads the model with the given name at the created directory.  """
        assert self.dir_load, "No loading directory found, initialize it before loading or run `create_save_directory`"
        dir_load = os.path.join(self.dir_load, self.checkpoint)
        save_path = self._get_files(dir_load=dir_load, load_name=load_name)
        return self._load_model(save_path)

    def _save_model(self, save_path: str, *args, **kwargs):
        """ Saves the actual model.  """
        if not os.path.exists(os.path.dirname(save_path)):
            raise NotImplementedError
        raise NotImplementedError

    def _load_model(self, load_path: str, *args, **kwargs) -> str:
        """ Loads the actual model, and return the load path.  """
        raise NotImplementedError

    @staticmethod
    def _current_time():
        return time.strftime('%Y-%b-%d-%a_%H.%M.%S')

    @staticmethod
    def _get_files(dir_load: str, load_name: str):
        """
        This returns the file location

        :param dir_load: Loading directory where the load name is going to be searched.
        :param load_name: Loading name of the file, or one of the keywords
            To retrieve the latest file: `last`, `newest`, `latest`
            To retrieve the oldest file: `first`, `oldest`
        """
        commands = dict(last=max, newest=max, latest=max, first=min, oldest=min)

        load_path = os.path.join(dir_load, load_name)
        if load_name in commands.keys():
            # Checking location on files.
            files = glob.glob(os.path.join(dir_load, "*"))
            if not len(files):
                raise ValueError(f"Unable to locate any models in `{dir_load}`")

            # Get the file depending on the load command
            load_path = commands[load_name](files, key=os.path.getctime)
        return load_path
