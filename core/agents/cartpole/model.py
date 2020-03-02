from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from core.models import BaseKerasModel


class CartPoleKerasModel(BaseKerasModel):

    @staticmethod
    def create_model(input_shape, output_shape):
        """ Creates the model.  """
        model = Sequential([
            Dense(input_shape=input_shape,
                  name='layer_fc1', units=24, activation='relu'),
            Dense(name='layer_fc2', units=24, activation='relu'),
            Dense(name='layer_fc_out', units=output_shape, activation='linear'),
        ])
        model.compile(optimizer=Adam(lr=1e-2), loss='mse')
        return model
