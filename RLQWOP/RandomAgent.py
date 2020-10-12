import numpy as np

class RandomAgent():
    def act(self, obs):
        return np.random.randint(2, size=4)