import numpy as np


class BaseMemoryNumpy:
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

        if stacked_frames:
            # States are expected to be LazyFrames
            self.states = [None for _ in range(size)]
            self.next_state = [None for _ in range(size)]
        else:
            self.states = np.zeros(shape=(size, *shape), dtype=np.float32)
            self.next_state = np.zeros(shape=(size, *shape), dtype=np.float32)

        self.actions = np.zeros(shape=size, dtype=np.uint8)
        self.rewards = np.zeros(shape=size, dtype=np.float16)
        self.done = np.zeros(shape=size, dtype=np.bool)

    def __getitem__(self, item):
        state = self.states[item]
        action = self.actions[item]
        reward = self.rewards[item]
        done = self.done[item]
        next_state = self.next_state[item]
        return state, action, reward, done, next_state

    def __len__(self):
        return self.pointer if not self.filled else self.size

    def is_full(self):
        return self.pointer == self.size

    def reset(self):
        self.pointer = 0
        self.filled = False

    def get_batch(self, batch_size=32):
        idx = np.random.randint(0, len(self) - 1, batch_size)

        if self.stacked_frames:
            states = [self.states[x] for x in idx]
            states_next = [self.next_state[x] for x in idx]
        else:
            states = self.states[idx]
            states_next = self.next_state[idx]

        actions = self.actions[idx]
        rewards = self.rewards[idx]
        done = self.done[idx]
        return states, actions, rewards, done, states_next

    def add(self, state, action, reward, done, next_state):
        if self.is_full():
            self.pointer = 0
            self.filled = True

        k = self.pointer
        self.pointer += 1

        self.states[k] = state
        self.actions[k] = action
        self.rewards[k] = reward
        self.done[k] = done
        self.next_state[k] = next_state
