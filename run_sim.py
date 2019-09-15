import time
import numpy as np
from hotrl.envs.house import House, Homie
from hotrl.envs.wrappers import FullyObsWrapper

homies = [Homie(initial_room='Bedroom')]
temperatures = np.array([
    [0, 0, 0, 0, 0],
    [0, 25, 25, 25, 0],
    [0, 30, 25, 25, 0],
    [0, 25, 25, 25, 0],
    [0, 0, 0, 0, 0]
], dtype=float)
env = FullyObsWrapper(House(size=5, homies=homies, temperatures=temperatures))

n_episodes = 10000
for n in range(n_episodes):
    time.sleep(0.2)
    obs, reward, done, info = env.step(0)
    object_matrix = obs[:, :, 0]
    temp_matrix = obs[:, :, 2]
    env.render(temperature=True)
    print(temp_matrix)
