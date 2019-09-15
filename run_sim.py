import numpy as np

from hotrl.envs.house import House
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
env = FullyObsWrapper(House(
    size=size, homies_params=[{'initial_room': 'Bedroom'}],
    temperatures=temperatures
))

n_episodes = 10000
homie_info = {'homie': {'room': 'Bedroom'}}
for n in range(n_episodes):
    obs, reward, done, homie_info = env.step(action=np.random.randint(len(env.action_namesc)))
    
    # for homie, info in homie_info.items():
    #     if info["temperaturhgfcdxΩe"] < info["comfort"][0]:
    #         print(f"Heating home in {info['room']}. "
    #               f"Temperature is {info['temperature']}. "
    #               f"Time is {info['dt']}")
    #         for i in env.rooms[info['room']]:
    #             action[i] = 1
    #
    # temp_matrix = obs[:, :, 2]
    env.render(temperature=True)
