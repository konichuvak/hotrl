## Inspiration
As part of the AI for Climate Change hackathon {TODO: link}, we wanted to develop an application of reinforcement learning that could help households in Montreal reduce their carbon footprint. This wasn't obvious, since so much of our household energy is low-carbon electricity from hydropower. However, residential heating still predominantly relies on fossil fuels while accounting for the lion's share of Canadians' energy consumption: making it an ideal problem to tackle for this hackathon. So, w built a simple 2D heat transfer model to simulate houses in Montreal's winter, and trained a reinforcement learning agent to optimize heating schedules so as to reduce waste and maximize the comfort of simulated occupants with their own schedule and temperature preferences.

## Environments
We created an openai gym environment for our simulations, which models a procedurally generated house as a 2D grid world along with it's heat transfer dynamics and the behavior of its occupants. The environments are inherited from ![gym-minigrid](https://github.com/maximecb/gym-minigrid) library to facilitate visualizations.

### House
Each cell represents a room in a 2-D grid. The cell can be occupied by one of the following objects:
`Homie` -- a person in the house
`Wall` -- a type of isolation that captures the heat 
`HeatingTile` -- a cell in the grid that can be targeted by the 
`Outdoors` -- a cell of the grid that can not be directly controlled by the system and assumes an outside temperature.

- TODO: insert picture here

### MultiRoomHouse
In contrast to the `House` environment, here a room may consist of multiple `HeatingTile`s surrounded by `Wall`s and connected via `Door`s with neighbouring rooms. This is a more realistic example, since we can assign various densities to the object, allowing for a more sophisticated heat transfer models.
Moreover, this allows to place multiple `Homie`s in the same room (with possibly different temperature preferences), adding even more interesting scenarios. 

- TODO: insert picture here


### Action Space
In both types of environemts, the actions space is represented by binary matrices shaped like the underlying grid, with active elements corresponding to the `HeatingTile`s that should be activated. To avoid controlling each tile individually actions are applied on each `Room` of the House. This is a simlification, but one could extend the heat transfer model to each individal room to achive more realistic simulation.

### Rewards
Two main sources of the reward are the discomfort_penalty and the cost of heating.

HSWQ
- TODO: how are the costs calculated?


# Temperature control


# Real-World Applications


## Existing heating systems
Heaters with knobs / buttons / sensors that could be manipulated.


## Newly-installed heating systems
Due to widespread connectivity, we expect modern heating systems to have internet connection.

## Cost savings analysis
