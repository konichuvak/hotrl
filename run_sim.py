import time
from gym_minigrid.wrappers import FullyObsWrapper

from hotrl.envs.house import House, Homie

n_episodes = 10000
homies = [Homie(initial_room='Bedroom')]
env = FullyObsWrapper(House(size=5, homies=homies))

for n in range(n_episodes):
    time.sleep(0.2)
    obs, reward, done, info = env.step(0)
    object_matrix = obs[:, :, 0]
    temp_matrix = obs[:, :, 1]
    env.render()

"""
added `House` object that describes environment with cells in the grid being rooms
added `Person` object that represents a tenant that can move around the rooms of the `House` and the position and preferences of which will determine the reward signal of the temperature controller
added `run_sim.py` script that runs a simulation
"""