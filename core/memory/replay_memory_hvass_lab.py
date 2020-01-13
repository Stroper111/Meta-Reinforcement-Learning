
import numpy as np


class ReplayMemoryHvassLab:
    """
        The replay-memory holds many previous states of the game-environment.
        This helps stabilize training of the Neural Network because the data
        is more diverse when sampled over thousands of different states.

        size: int
            Capacity of the replay-memory. This is the number of states.
        shape: Union[tuple, list]
            The dimensions of an observation
        action_space: int
            Number of possible actions in the game-environment.
        alpha: float
            Learning rate used for updating Q-values
        gamma: float
            Discount-factor used for updating Q-values.
    """

    def __init__(self, size, shape, action_space, alpha=0.10, gamma=0.97, stackedframes=False):
        self.size = size
        self.shape = shape
        self.alpha =  alpha
        self.gamma = gamma

        self.pointer = 0
        self.filled = False
        self.error_threshold = 0.1

        if stackedframes:
            # States are expected to be LazyFrames
            self.states = [None for _ in range(size)]
        else:
            self.states = np.zeros(shape=(size, *shape), dtype=np.float32)
        self.q_values = np.zeros(shape=(size, action_space), dtype=np.float32)
        self.q_values_old = np.zeros(shape=(size, action_space), dtype=np.float32)
        self.actions = np.zeros(shape=size, dtype=np.uint8)
        self.rewards = np.zeros(shape=size, dtype=np.float16)
        self.end_life = np.zeros(shape=size, dtype=np.bool)
        self.end_episode = np.zeros(shape=size, dtype=np.bool)
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

    def __getitem__(self, item):
        state = self.states[item]
        q_values = self.q_values[item]
        action = self.actions[item]
        reward = self.rewards[item]
        end_life = self.end_life[item]
        end_episode = self.end_episode[item]
        return state, q_values, action, reward, end_life, end_episode

    def __len__(self):
        return self.pointer if not self.filled else self.size

    def get_batch(self, idx):
        states = [self.states[x] for x in idx]
        return states, self.q_values[idx]

    def reset(self):
        self.pointer = 0
        self.filled = False

    def refill_memory(self):
        self.pointer = 0
        self.filled = True

    def pointer_ratio(self):
        return self.pointer / self.size

    def is_full(self):
        return self.pointer >= self.size

    def add(self, state, q_values, action, reward, end_episode):
        if not self.is_full():
            k = self.pointer
            self.pointer += 1

            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_episode[k] = end_episode
            self.rewards[k] = np.clip(reward, -1.0, 1.0)

    def update(self):
        self.q_values_old[:] = self.q_values[:]
        for k in reversed(range(self.pointer - 1)):
            action = self.actions[k]
            reward = self.rewards[k]
            end_episode = self.end_episode[k]

            if end_episode:
                action_value = reward
            else:
                action_value = reward + self.gamma * np.max(self.q_values[k + 1])

            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])
            self.q_values[k, action] = action_value

    def print_statistics(self):
        print("\nReplay-memory statistics")

        msg = "\tQ-values Before, Min: {min:5.2f}, Mean: {mean:5.2f}, Max: {max:5.2f}"
        print(msg.format(min=np.min(self.q_values_old), mean=np.mean(self.q_values_old), max=np.max(self.q_values_old)))

        msg = "\tQ-values After, Min: {min:5.2f}, Mean: {mean:5.2f}, Max: {max:5.2f}"
        print(msg.format(min=np.min(self.q_values), mean=np.mean(self.q_values), max=np.max(self.q_values)))

        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff., Min: {min:5.2f}, Mean: {mean:5.2f}, Max: {max:5.2f}"
        print(msg.format(min=np.min(q_dif), mean=np.mean(q_dif), max=np.max(q_dif)))

        # Drop last one, since it hasn't been updated yet.
        error = self.estimation_errors[:-1]
        error_count = np.count_nonzero(error > self.error_threshold)
        msg = "\tNumber of large errors > {threshold}: {error_count}/{total} ({percentage:.1%})"
        print(msg.format(threshold=self.error_threshold, error_count=error_count, total=self.pointer,
                         percentage=error_count / self.pointer))

        end_episode_percentage = np.count_nonzero(self.end_episode) / self.pointer
        non_zero_percentage = np.count_nonzero(self.rewards) / self.pointer
        msg = "\tend_episode: {episode:.1%}, reward non-zero: {non_zero:.1%}"
        print(msg.format(episode=end_episode_percentage, non_zero=non_zero_percentage))
