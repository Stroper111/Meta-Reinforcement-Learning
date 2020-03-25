import keras

from keras.models import Sequential
from keras.layers import Embedding, Reshape, Dense
from keras.optimizers import Adam

from core.models.extern import BaseModelKeras


class TaxiModel(BaseModelKeras):
    @staticmethod
    def create_model(input_shape, output_shape, *args, **kwargs):
        model = Sequential()
        model.add(Embedding(500, 6, input_length=input_shape))
        model.add(Reshape((6,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(output_shape, activation='linear'))
        model.compile(optimizer=Adam(lr=1e-2), loss='mae')
        return model


class TaxiModelEmbedding(BaseModelKeras):
    @staticmethod
    def create_model(input_shape, output_shape, *args, **kwargs):
        model = Sequential()
        model.add(Embedding(500, 6, input_length=input_shape))
        model.add(Reshape((6 * input_shape,)))
        model.compile(Adam(lr=1e-2), loss='mae', metrics=['mae'])
        return model
