import numpy as np


class BaseReplayMemory:
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
            This is available in case you want to extend it with q values
    """

    def __init__(self, size, shape, action_space=None, stacked_frames=False):
        self.size = size
        self.shape = shape
        self.action_space = action_space
        self.stacked_frames = stacked_frames

        self.pointer = 0
        self.filled = False
        self.error_threshold = 0.1

        if stacked_frames:
            # States are expected to be LazyFrames
            self.states = [None for _ in range(size)]
        else:
            self.states = np.zeros(shape=(size, *shape), dtype=np.float32)

        self.actions = np.zeros(shape=size, dtype=np.uint8)
        self.rewards = np.zeros(shape=size, dtype=np.float16)
        self.end_episode = np.zeros(shape=size, dtype=np.bool)
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

    def __getitem__(self, item):
        state = self.states[item]
        action = self.actions[item]
        reward = self.rewards[item]
        end_episode = self.end_episode[item]
        return state, action, reward, end_episode

    def __len__(self):
        return self.pointer if not self.filled else self.size

    def get_batch(self, idx):
        if self.stacked_frames:
            states = [self.states[x] for x in idx]
            states_next = [self.states[x + 1] for x in idx]
        else:
            states = self.states[idx]
            states_next = self.states[idx]

        actions = self.actions[idx]
        rewards = self.rewards[idx]
        done = self.end_episode[idx]
        return states, actions, rewards, done, states_next

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

    def add(self, state, action, reward, end_episode):
        if not self.is_full():
            k = self.pointer
            self.pointer += 1

            self.states[k] = state
            self.actions[k] = action
            self.end_episode[k] = end_episode
            self.rewards[k] = np.clip(reward, -1.0, 1.0)
