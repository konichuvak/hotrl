import time
import numpy as np

from hotrl.envs.house import House
from hotrl.envs.house import House, Homie, MultiRoomHouse
from hotrl.envs.wrappers import FullyObsWrapper

size = 4
inside_temp = 15.
outside_temp = 5.
temperatures = np.pad(
    np.full((size-2, size-2), fill_value=inside_temp),
    pad_width=[(1, 1), (1, 1)],
    mode='constant',
    constant_values=outside_temp
)
# env = FullyObsWrapper(House(
#     size=size, homies_params=[{'initial_room': 'Bedroom'}],
#     temperatures=temperatures
# ))
env = FullyObsWrapper(MultiRoomHouse(homies=homies, seed=np.random.randint(0,100)))


n_episodes = 10000
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
