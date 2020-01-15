import numpy as np
import sys
import keras.backend as K

from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.initializers import TruncatedNormal
from keras.optimizers import RMSprop
from collections import deque

from core.models import BaseModel


class HvassLab(BaseModel):
    def __init__(self, input_shape, action_space, epsilon=0.05):
        self.input_shape = input_shape
        self.action_space = action_space

        self.back_up_count = 2
        self.save_msg = "episode {:7,d} frames {:11,d} {:s}"

        # This is a hack to update the learning rate with a functions
        # In hvasslab they use tf, which has an easier interface (for a change)
        self.optimizer = RMSprop(learning_rate=1e-3)

        self.model = self.create_model(input_shape, action_space)
        self.epsilon = epsilon

    @staticmethod
    def create_model(input_shape, output_shape):
        init = TruncatedNormal(mean=0, stddev=2e-2)

        model = Sequential(
            [
                Conv2D(input_shape=input_shape, name='layer_conv1',
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
                Dense(name='layer_fc_out', units=output_shape,
                      kernel_initializer=init, activation='linear')
            ]
        )
        model.compile(optimizer=RMSprop(lr=1e-4), loss='mse')
        # model.summary()
        return model

    def actions(self, states):
        """ For this example we sample epsilon outside of the model.  """
        q_values = self.model.predict(states)
        return q_values

    def optimize(self, replay_memory, min_epochs=1., max_epochs=10, batch_size=128, loss_limit=0.015,
                 learning_rate=1e-3):
        print("\nOptimizing Neural Network to better estimate Q-values ...")
        print("\tLearning-rate: %.1e" % learning_rate)
        print("\tLoss-limit: %.3f" % loss_limit)
        print("\tMin epochs: %.1f" % min_epochs)
        print("\tMax epochs: %.1f" % max_epochs)

        replay_memory.prepare_sampling(batch_size=batch_size)
        iterations_per_epoch = replay_memory.pointer / batch_size

        iterations_min = int(iterations_per_epoch * min_epochs)
        iterations_max = int(iterations_per_epoch * max_epochs)
        loss_history = deque(maxlen=100)

        for iteration in range(iterations_max):
            batch_states, batch_q_values = replay_memory.batch_random()
            loss = self.model.fit(batch_states, batch_q_values, verbose=0).history['loss'][0]
            loss_history.append(loss)
            loss_mean = sum(loss_history) / len(loss_history)

            # TODO put this back after server
            if iteration > iterations_min and loss_mean < loss_limit:
                percentage_epoch = iteration / iterations_per_epoch
                msg = "\r\tIteration: {iteration} ({pct:.2f} epoch), Batch loss: {loss:.4f}, Mean loss: {mean_loss:.4f}"
                sys.stdout.write(msg.format(iteration=iteration, pct=percentage_epoch, loss=loss, mean_loss=loss_mean))
                sys.stdout.flush()
                break

        print("\n")

    def change_learning_rate(self, learning_rate):
        K.set_value(self.optimizer.lr, learning_rate)
