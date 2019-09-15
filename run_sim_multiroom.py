import time
import numpy as np
from hotrl.envs.house import House, Homie, MultiRoomHouse
from hotrl.envs.wrappers import FullyObsWrapper

homies = [Homie(initial_room='Bedroom')]
temperatures = np.array([
    [-30, -30, -30, -30, -30],
    [-30, 25, 25, 25, -30],
    [-30, 25, 25, 25, -30],
    [-30, 25, 25, 25, -30],
    [-30, -30, -30, -30, -30]
], dtype=float)
# env = FullyObsWrapper(House(size=5, homies=homies, temperatures=temperatures))
env = FullyObsWrapper(MultiRoomHouse(homies=homies, seed=np.random.randint(0,100)))

n_episodes = 10000
action = [0,1,4]
for n in range(n_episodes):
    obs, reward, done, homie_info = env.step(action)
    action = []
    time.sleep(0.1)
    for homie, info in homie_info.items():
        if info["temperature"] < info["comfort"][1]:
            print(f"Heating home in {info['room']}. "
                  f"Temperature is {info['temperature']}. "
                  f"Time is {info['dt']}")
            if info['room'] == 'Outside':
                continue
            action.append(env.room_names.index(info['room']))
    object_matrix = obs[:, :, 0]
    temp_matrix = obs[:, :, 2]
    env.render(temperature=True)
