import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from core.models import BaseModel


class BaseModelGym(BaseModel):
    @staticmethod
    def create_model(input_shape, action_space, *args, **kwargs):
        model = Sequential(
            [
                Dense(name='layer_fc_in', input_shape=input_shape, units=24, activation='relu'),
                Dense(name='layer_fc1', units=24, activation='relu'),
                Dense(name='layer_fc_out', units=action_space, activation='linear')
            ]
        )
        model.compile(optimizer=Adam(lr=1e-3), loss='mse')
        # model.summary()
        return model

    def actions(self, states):
        """ For this example we sample epsilon outside of the model.  """
        q_values = self.model.predict(states)
        print(q_values.shape)
        action = np.argmax(q_values, axis=1)
        return q_values, action
