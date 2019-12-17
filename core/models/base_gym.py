from keras import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

from core.models import BaseModel


class BaseModelGym(BaseModel):
    @staticmethod
    def create_model(input_shape, action_space):
        model = Sequential(
                [
                    Dense(name='layer_fc_in', input_shape=(4,), units=16, activation='relu'),
                    Dense(name='layer_fc1', units=32, activation='relu'),
                    Dense(name='layer_fc2', units=32, activation='relu'),
                    Dense(name='layer_fc3', units=16, activation='relu'),
                    Dense(name='layer_fc_out', units=action_space, activation='linear')
                ]
        )
        model.compile(optimizer=RMSprop(lr=0.0025), loss='mse')
        model.summary()
        return model
