
import numpy as np


class LinearControlSignal:
    """
    A control signal that changes linearly over time.

    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""
        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value
        return value


class StepWiseSignal:
    def __init__(self, start_value, end_value, num_iterations, bins, repeat=False):
        self.start_value = start_value
        self.end_Value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        self.bins = bins
        self.bin_size = num_iterations // bins
        self.step_size = (self.end_Value - self.start_value) / bins

    def get_value(self, iteration):
        if self.repeat:
            iteration %= self.num_iterations

        bin = iteration // self.bin_size
        return min(self.start_value + bin * self.step_size, self.end_Value)


class EpsilonGreedy:
    """
        The epsilon-greedy policy either takes a random action with
        probability epsilon, or it takes the action for the highest
        Q-value.

        If epsilon is 1.0 then the actions are always random.
        If epsilon is 0.0 then the actions are always argmax for the Q-values.

        Epsilon is typically decreased linearly from 1.0 to 0.1
        and this is also implemented in this class.

        During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions, epsilon_testing=0.05, num_iterations=1e6,
                 start_value=1.0, end_value=0.1, repeat=False):
        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, steps, training):
        """
            Return the epsilon for the given iteration.
            If training==True then epsilon is linearly decreased,
            otherwise epsilon is a fixed number.
        """
        if training:
            epsilon = self.epsilon_linear.get_value(iteration=steps)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training):
        epsilon = self.get_epsilon(steps=iteration, training=training)

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon
