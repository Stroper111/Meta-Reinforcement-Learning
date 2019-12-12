import os
import glob
import logging
import numpy as np

from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.initializers import TruncatedNormal
from keras.optimizers import RMSprop

from core.models import AbstractModel


class BaseModel(AbstractModel):
    def __init__(self, input_shape, action_space, epsilon=0.05):
        super().__init__(input_shape, action_space)

        self.input_shape = input_shape
        self.action_space = action_space

        self.back_up_count = 2
        self.save_msg = "episode {:6,d} frames {:11,d} {:s}"

        self.model = self.create_model(input_shape, action_space)
        self.epsilon = epsilon

    @staticmethod
    def create_model(input_shape, action_space):
        init = TruncatedNormal(mean=0, stddev=2e-2)

        model = Sequential(
            [
                Conv2D(input_shape=input_shape, name='layer_conv1',
                       filters=16, kernel_size=3, strides=2,
                       padding='same', kernel_initializer=init,
                       activation='relu'),
                Conv2D(input_shape=input_shape, name='layer_conv2',
                       filters=32, kernel_size=3, strides=2,
                       padding='same', kernel_initializer=init,
                       activation='relu'),
                Flatten(),
                Dense(name='layer_fc2', units=256, activation='relu'),
                Dense(name='layer_fc3', units=256, activation='relu'),
                Dense(name='layer_fc_out', units=action_space, activation='linear')
            ]
        )
        model.compile(optimizer=RMSprop(lr=0.0025), loss='mse')
        model.summary()
        return model

    def predict(self, states):
        return self.model.predict(states)

    def actions(self, states):
        q_values = self.model.predict(states)
        random = np.random.random(len(q_values))
        explore = np.where(random < self.epsilon)

        actions_explore = np.random.randint(low=0, high=self.action_space, size=len(states))
        actions_exploit = np.argmax(q_values, axis=1)

        if explore:
            actions_exploit[explore] = actions_explore[explore]
        return q_values, actions_exploit

    def train(self, sampling):
        loss_history = []
        for num, (x, y) in enumerate(sampling):
            loss = self.model.fit(x, y, verbose=0).history['loss'][0]
            loss_history.append(loss)
            print("\r\tIteration {:4,d}/{:4,d}, batch_loss: {:7,.4f}".format(num, len(sampling), loss), end='')
        return loss_history

    def save_model(self, save_dir, episode, frames):
        self._check_create_directory(save_dir)
        save_file = os.path.join(save_dir, self.save_msg.format(episode, frames, "weights.h5"))
        self.model.save_weights(save_file)

    def save_checkpoint(self, save_dir, episode, frames):
        self.save_model(save_dir, episode, frames)
        self._remove_old_files(pattern="*weights.h5", save_dir=save_dir, back_ups=self.back_up_count)

    def load_model(self, load_dir):
        """ Automatically load the newest model it can find in a sub directory.  """
        files = glob.glob(os.path.join(load_dir, "..", "/*/", "*weights.h5"))
        if files:
            newest_file = max(files, key=os.path.getctime)
            self.model.load_weights(newest_file)
            print("Checkpoint found, continuing...")
        else:
            print("No checkpoint found, reinitializing variables instead...")

    def load_checkpoint(self, load_dir):
        self.load_model(load_dir)

    def _check_create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        return self

    @staticmethod
    def _remove_old_files(pattern, save_dir, back_ups):
        """ Helper to store only a limited amount of backups.  """
        files = glob.glob(os.path.join(save_dir, pattern))
        if len(files) >= (back_ups + 1):
            oldest_file = min(files, key=os.path.getctime)
            os.remove(oldest_file)
            logging.info(f"Os removed:\n\t{oldest_file}")
