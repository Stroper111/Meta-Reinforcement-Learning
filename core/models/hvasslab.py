import numpy as np

from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.initializers import TruncatedNormal
from keras.optimizers import RMSprop

from core.models import BaseModel


class HvassLab(BaseModel):
    def __init__(self, input_shape, action_space, epsilon=0.05):
        super().__init__(input_shape, action_space, epsilon)

    @staticmethod
    def create_model(input_shape, output_shape):
        init = TruncatedNormal(mean=0, stddev=2e-2)

        model = Sequential(
            [
                Conv2D(input_shape=output_shape, name='layer_conv1',
                       filters=16, kernel_size=3, strides=2,
                       padding='same', kernel_initializer=init,
                       activation='relu'),

                Conv2D(name='layer_conv2',
                       filters=32, kernel_size=3, strides=2,
                       padding='same', kernel_initializer=init,
                       activation='relu'),

                Conv2D(name='layer_conv3',
                       filters=64, kernel_size=3, strides=1,
                       padding='same', kernel_initializer=init,
                       activation='relu'),

                Flatten(),

                Dense(name='layer_fc1', units=1024,
                      kernel_initializer=init, activation='relu'),

                Dense(name='layer_fc2', units=1024,
                      kernel_initializer=init, activation='relu'),

                Dense(name='layer_fc3', units=1024,
                      kernel_initializer=init, activation='relu'),

                Dense(name='layer_fc4', units=1024,
                      kernel_initializer=init, activation='relu'),

                # Linear is important for Q-values!
                Dense(name='layer_fc_out', units=input_shape,
                      kernel_initializer=init, activation='linear')
            ]
        )
        model.compile(optimizer=RMSprop(lr=1e-4), loss='mse')
        # model.summary()
        return model

    def actions(self, states):
        """ For this example we sample epsilon outside of the model.  """
        q_values = self.model.predict(states)
        action = np.argmax(q_values, axis=1)
        return q_values, action
