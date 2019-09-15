import time
import numpy as np
from hotrl.envs.house import House, Homie
from hotrl.envs.wrappers import FullyObsWrapper

homies = [Homie(initial_room='Bedroom')]
temperatures = np.array([
    [-30, -30, -30, -30, -30],
    [-30, 25, 25, 25, -30],
    [-30, 25, 25, 25, -30],
    [-30, 25, 25, 25, -30],
    [-30, -30, -30, -30, -30]
], dtype=float)
env = FullyObsWrapper(House(size=5, homies=homies, temperatures=temperatures))

n_episodes = 10000
action = np.zeros((5, 5))
for n in range(n_episodes):
    obs, reward, done, homie_info = env.step(action)
    for homie, info in homie_info.items():
        if info["temperature"] < info["comfort"][0]:
            print(f"Heating home in {info['room']}. "
                  f"Temperature is {info['temperature']}. "
                  f"Time is {info['dt']}")
            for i in env.rooms[info['room']]:
                action[i] = 1

    object_matrix = obs[:, :, 0]
    temp_matrix = obs[:, :, 2]
    env.render(temperature=True)
