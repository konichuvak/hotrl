import numpy as np

import gym
from gym import spaces


class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=-50,
            high=50,
            shape=(self.env.width, self.env.height, 2),
            dtype='int8'
        )

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        
        # Remove the dimension corresponding to color of the objects
        full_grid = np.delete(full_grid, 1, 2)
        return full_grid