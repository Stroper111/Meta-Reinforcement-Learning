from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam

from core.models.extern import BaseModelKeras


class DQNModel(BaseModelKeras):

    @staticmethod
    def create_model(input_shape, output_shape, *args, **kwargs):
        """ Creates the model.  """
        model = Sequential([
            Dense(input_shape=input_shape,
                  name='layer_fc1', units=256, activation='relu'),
            Dense(name='layer_fc2', units=512, activation='relu'),
            Dense(name='layer_fc3', units=512, activation='relu'),
            Dense(name='layer_fc4', units=256, activation='relu'),
            Dense(name='layer_fc5', units=128, activation='relu'),
            Dense(name='layer_fc_out', units=output_shape, activation='linear'),
        ])
        model.compile(optimizer=Adam(lr=1e-2), loss='mse')
        return model
