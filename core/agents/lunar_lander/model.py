import numpy as np

from core.models import AbstractModel

from keras import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from keras.models import load_model


class LunarLanderModel(AbstractModel):
    def __init__(self, input_shape: tuple, output_shape: int):
        super().__init__(input_shape, output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self.create_model(input_shape, output_shape)

    def create_model(self, input_shape: tuple, output_shape: int, *args, **kwargs):
        """
        Compiles a fully-connected neural network model
        :param input_shape: A Tuple describing the input to the network ((8,) for Lunar Lander)
        :param output_shape: An Integer describing the action_space (4 for Lunar Lander)
        :return: A Keras Model object
        """
        model = Sequential()
        model.add(Dense(150,
                        input_shape=input_shape,
                        kernel_initializer='he_normal',
                        bias_initializer='zeros'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(120,
                        kernel_initializer='he_normal',
                        bias_initializer='zeros'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def predict(self, states):
        """ Return the predictions of your model.  """
        return self.model.predict(states)

    def actions(self, states):
        """ Return the argmax actions to execute following an epsilon-greedy strategy """
        q_values = self.model.predict(states)
        return np.argmax(q_values, axis=1)

    def actions_random(self, shape: tuple):
        """
        Return an array of random actions based on a given shape (usually (1,))

        :param shape: The amount of random actions you want to return
        :return: A NumPy Array containing random actions (integers)
        """
        return np.random.randint(0, self.output_shape, shape)

    def train(self, x, y):
        """ Trains a model. """
        return self.model.fit(x, y, verbose=0)

    def save_model(self, save_dir: str, episode: int, steps="_"):
        """ Store the model to a file."""
        self.model.save(f"{save_dir}/lunar_lander_{episode}_model.h5")

    def save_checkpoint(self, save_dir: str, episode: int, steps="_"):
        """ Store the model and metadata to continue a training session. """
        self.model.save_weights(f"{save_dir}/lunar_lander_{episode}_weights.h5")

    def load_model(self, load_dir: str):
        """
        Load a keras model.
        :return: A Keras Model object
        """
        return load_model(load_dir)

    def load_checkpoint(self, load_dir: str):
        """
        Continues a training run from the checkpoint.
        :return: A Keras Model object
        """
        return self.model.load_weights(load_dir)
