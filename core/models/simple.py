import os
import glob
import logging

from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.initializers import TruncatedNormal
from keras.optimizers import RMSprop

from collections import deque

from core.models import AbstractModel


class BaseModel(AbstractModel):
    def __init__(self, input_shape, action_space):
        super().__init__(input_shape, action_space)

        self.input_shape = input_shape
        self.action_space = action_space

        self.back_up_count = 2
        self.episodes = 0
        self.frames = 0
        self.save_msg = "episode {:6,d} frames {:11,d} {:s}"

        self.model = self.create_model(input_shape, action_space)

    @staticmethod
    def create_model(input_shape, action_space):
        init = TruncatedNormal(mean=0, stddev=2e-2)

        model = Sequential(
            [
                Conv2D(input_shape=input_shape, name='layer_conv1',
                       filters=16, kernel_size=3, strides=2,
                       padding='same', kernel_initializer=init,
                       activation='relu'),

                Flatten(),

                Dense(name='layer_fc_out', units=action_space,
                      activation='linear')
            ]
        )
        model.compile(optimizer=RMSprop(lr=0.0025), loss='mse')
        model.summary()
        return model

    def predict(self, states):
        return self.model.predict(states)

    def train(self, sampling):
        loss = self.model.fit(*sampling.random_batch(), verbose=0).history['loss']
        return loss

    def save_model(self, save_dir):
        self._check_create_directory(save_dir)
        save_file = os.path.join(save_dir, self.save_msg.format(self.episodes, self.frames, "weights.h5"))
        self.model.save_weights(save_file)

    def save_checkpoint(self, save_dir):
        self.save_model(save_dir)
        self._remove_old_files(pattern="*weights.h5", save_dir=save_dir, back_ups=self.back_up_count)

    def load_model(self, load_dir):
        files = glob.glob(os.path.join(load_dir, "*weights.h5"))
        newest_file = max(files, key=os.path.getctime)
        self.model.load_weights(newest_file)

    def load_checkpoint(self, load_dir):
        self.load_model(load_dir)

    def _check_create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return self

    def _remove_old_files(self, pattern, save_dir, back_ups):
        """ Helper to store only a limited amount of backups.  """
        files = glob.glob(os.path.join(save_dir, pattern))
        if len(files) >= (back_ups + 1):
            oldest_file = min(files, key=os.path.getctime)
            os.remove(oldest_file)
            logging.info(f"Os removed:\n\t{oldest_file}")
