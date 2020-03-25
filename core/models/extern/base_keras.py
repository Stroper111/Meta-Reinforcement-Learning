import numpy as np
import pathlib
import time
import os
import glob

# ignore keras future warning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import keras

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

from core.models import BaseModel


class BaseModelKeras(BaseModel):
    checkpoint: str = "checkpoints"
    dir_model: str = "pretrained"
    dir_load: str = None
    dir_save: str = None

    model: keras.models

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.model = self.create_model(input_shape, output_shape)

    @staticmethod
    def create_model(input_shape, output_shape, *args, **kwargs):
        """ Creates the model.  """
        model = Sequential([
            Dense(input_shape=input_shape,
                  name='layer_fc1', units=24, activation='relu'),
            Dense(name='layer_fc2', units=24, activation='relu'),
            Dense(name='layer_fc_out', units=output_shape, activation='linear'),
        ])
        model.compile(optimizer=Adam(lr=1e-2), loss='mse')
        return model

    def predict(self, states):
        """ Return the predictions of your model.  """
        return self.model.predict(states)

    def actions(self, states):
        """ Return the actions to execute, this can be combined with epsilon.  """
        action = np.argmax(self.predict(states), axis=1)
        return np.array(action)

    def actions_random(self, shape: tuple):
        return np.random.randint(0, self.output_shape, shape)

    def train(self, x, y):
        """ Trains a model. """
        return self.model.fit(x, y, verbose=0)

    def save_model(self, save_name: str, weights_only: bool = True, *args, **kwargs):
        """ This stores the model with given save name at the created directory.  """
        assert self.dir_save, "No saving directory found, initialize it before loading or run `create_save_directory`"
        save_path = os.path.join(self.dir_save, save_name)
        self._save_model(save_path, weights_only)

    def load_model(self, load_name: str = 'last', weights_only: bool = True, *args, **kwargs):
        """
        This loads a model at a specific time,

        :param model: the model that needs to be saved
        :param weights_only: Extra option for keras/tensorflow models.
        :param load_name: Name of the model to load options are the exact name of the model, 'last' and 'first'.
        """
        assert self.dir_load, "No loading directory found, initialize it before loading or run `create_save_directory`"
        load_path = self._get_files(dir_load=self.dir_load, load_name=load_name)
        self._load_model(load_path, weights_only)

    def save_checkpoint(self, save_name: str, weights_only: bool = True, *args, **kwargs):
        """ This stores the model with given save name at the created directory.  """
        assert self.dir_save, "No saving directory found, initialize it before loading or run `create_save_directory`"
        save_path = os.path.join(self.dir_save, self.checkpoint, save_name)
        self._save_model(save_path, weights_only)

    def load_checkpoint(self, load_name: str = 'last', weights_only: bool = True, *args, **kwargs):
        """ This loads the model with the given name at the created directory.  """
        assert self.dir_load, "No loading directory found, initialize it before loading or run `create_save_directory`"
        dir_load = os.path.join(self.dir_load, self.checkpoint)
        save_path = self._get_files(dir_load=dir_load, load_name=load_name)
        self._load_model(save_path, weights_only)

    def _save_model(self, save_path: str, *weights_only: bool):
        """ Saves the actual model.  """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if weights_only:
            self.model.save_weights(save_path)
        else:
            self.model.save(save_path)

    def _load_model(self, load_path: str, *weights_only: bool):
        """ Loads the actual model.  """
        if weights_only:
            self.model.load_weights(load_path)
        else:
            self.model = load_model(load_path)
