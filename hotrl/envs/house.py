from datetime import datetime, timedelta
from typing import Tuple, List, Dict

import numpy as np
import termcolor
from gym import spaces

from gym_minigrid.envs.empty import MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX,\
    TEMPERATURES, COLORS
from gym_minigrid.minigrid import WorldObj, CELL_PIXELS, Grid, Floor
from hotrl.heat_transfer.model import SimpleModel

xy_coord = Tuple[int, int]
RoomType = ['Kitchen', 'Bathroom', 'Bedroom', 'LivingRoom', 'Outside']


class HeatingTile(Floor):
    
    def __init__(self, color='blue', temperature=20):
        super().__init__(color, temperature)


class Homie(WorldObj):
    
    def __init__(self, house: 'House', initial_room: RoomType = 'Bedroom'):
        super(Homie, self).__init__('ball', color='blue')
        self.house = house
        self.current_room = initial_room
        self.cur_pos = self._place_within_the_room()
    
    def _place_within_the_room(self) -> xy_coord:
        current_room = self.house.rooms[self.current_room]
        return current_room[0]
    
    def get_preferred_temperature(self, timestamp: datetime = None) -> int:
        """ Query homie for a preferred temperature at a current location,
        at a given time of the day or season"""
        
        if self.current_room == 'Kitchen':
            temp = 20, 25
        elif self.current_room == 'Bathroom':
            temp = 22, 24
        elif self.current_room == 'Bedroom':
            temp = 18, 20
        elif self.current_room == 'LivingRoom':
            temp = 19, 24
        elif self.current_room == 'Outside':
            temp = -30, 30
        else:
            raise ValueError('Undefined room type')
        
        return temp
    
    def step(self, timestamp: datetime) -> RoomType:
        date = dict(
            year=timestamp.year,
            month=timestamp.month,
            day=timestamp.day
        )
        
        sleep = datetime(**dict(**date, hour=0))
        morning_bath = datetime(**dict(**date, hour=7))
        breakfast = datetime(**dict(**date, hour=7, minute=30))
        leave_for_work = datetime(**dict(**date, hour=8))
        dinner = datetime(**dict(**date, hour=18))
        study = datetime(**dict(**date, hour=19))
        evening_bath = datetime(**dict(**date, hour=23, minute=30))
        
        if sleep <= timestamp < morning_bath:
            self.current_room = 'Bedroom'
        elif morning_bath <= timestamp < breakfast:
            self.current_room = 'Bathroom'
        elif breakfast <= timestamp < leave_for_work:
            self.current_room = 'Kitchen'
        elif leave_for_work <= timestamp < dinner:
            self.current_room = 'Outside'
        elif dinner <= timestamp < study:
            self.current_room = 'Kitchen'
        elif study <= timestamp < evening_bath:
            self.current_room = 'LivingRoom'
        elif evening_bath <= timestamp:
            self.current_room = 'Bathroom'
        
        self.cur_pos = self._place_within_the_room()
    
    def render(self, r, temperature: bool = False):
        if not temperature:
            self._set_color(r, temperature)
            r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)
        else:
            c = TEMPERATURES[self.temperature] if temperature else COLORS[
                self.color]
            r.setColor(*c)
            r.drawPolygon([
                (1, CELL_PIXELS),
                (CELL_PIXELS, CELL_PIXELS),
                (CELL_PIXELS, 1),
                (1, 1)
            ])


class HouseGrid(Grid):
    """ A grid-world house with tenants and temperatures for each cell
    """
    
    def encode(self, vis_mask=None):
        """ Produce a compact numpy encoding of the grid
        """
        
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)
        
        array = np.zeros((self.width, self.height, 3), dtype='int8')
        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        assert ValueError
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                    else:
                        array[i, j, 0] = OBJECT_TO_IDX[v.type]
                        array[i, j, 1] = COLOR_TO_IDX[v.color]
                        array[i, j, 2] = v.temperature
        
        return array


class House(MiniGridEnv):
    
    def __init__(
        self,
        temperatures: np.ndarray,
        size: int = 4,
        start_dt: datetime = datetime.now(),
        dt_delta: timedelta = timedelta(minutes=1),
        homies_params: List[Dict] = None,
        homie_reward_scaler: float = 2,
        action_threshold: float = 0.5,
    ):
        self.rooms = {
            'Kitchen'   : [(2, 2)],
            'Bathroom'  : [(2, 1)],
            'Bedroom'   : [(1, 2)],
            'LivingRoom': [(1, 1)],
            'Outside'   : [(0, i) for i in range(size)] +
                          [(i, 0) for i in range(size)] +
                          [(size - 1, i) for i in range(size)] +
                          [(i, size - 1) for i in range(size)]
        }
        self.homies = [Homie(self, **params) for params in homies_params]
        self.temperatures = temperatures
        self.current_dt = start_dt - timedelta(
            seconds=start_dt.second,
            microseconds=start_dt.microsecond
        )
        self.timedelta = dt_delta
        self.homie_reward_scaler = homie_reward_scaler
        self.action_threshold = action_threshold
        self.model = SimpleModel(heater_output=1000)
        
        n = 1440
        signal = np.cos(np.pi * np.arange(n) / float(n/2))
        self.daily_weather = self.rescale_linear(signal, 20, 10)
        
        super().__init__(
            grid_size=4,
            max_steps=1000,
            see_through_walls=True,
        )
        
        # Actions are discrete integer values corresponding to the combinations
        # of rooms that we want to start heating up
        self.action_matrices = [np.zeros((size, size))]
        self.action_names = ['heat_Nothing']
        for room in sorted(self.rooms):
            if room == 'Outside':
                continue
            heatmap = np.zeros((size, size))
            heatmap[list(zip(*self.rooms[room]))] = 1
            self.action_names.append(f'heat_{room}')
            self.action_matrices.append(heatmap.copy())
        
        self.action_space = spaces.Discrete(len(self.action_names))
    
    @staticmethod
    def rescale_linear(array, new_min, new_max):
        """Rescale an arrary linearly."""
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return m * array + b
    
    def reset(self):
        super().reset()
        for homie in self.homies:
            homie.step(self.current_dt)
    
    def _gen_grid(self, width, height):
        assert width == height
        
        # Create an empty grid
        self.grid = HouseGrid(width, height)
        
        # Generate walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place tenants in the house
        for homie in self.homies:
            self.grid.set(*homie.cur_pos, v=homie)
        
        self.place_agent()
        
        # Place the heating tiles in the house
        for i, cell in enumerate(self.grid.grid):
            x, y = divmod(i, width)
            if cell is None:
                self.grid.grid[i] = HeatingTile(
                    temperature=self.temperatures[x, y]
                )
            self.grid.grid[i].temperature = self.temperatures[x, y]
        
        self.mission = "it's getting hot in here"
    
    def change_temperature(self, heatmap: np.ndarray = None):
        """ Changes the temperature of each object in the house """
        if heatmap is None:
            heatmap = np.zeros(self.temperatures.shape, dtype=float)
        self.temperatures = self.model.step(
            temperatures=self.temperatures,
            heat=heatmap
        )
        for i, cell in enumerate(self.grid.grid):
            x, y = divmod(i, self.grid.width)
            self.grid.grid[i].temperature = self.temperatures[x, y]
    
    def step(self, action):
        self.current_dt += self.timedelta
        self.step_count += 1
        reward = 0
        
        # Reset the temperature outside of the house
        idx = self.current_dt.hour * 60 + self.current_dt.minute
        self.temperatures[list(zip(*self.rooms['Outside']))] = self.daily_weather[idx]
        
        # Move each homie and determine their preference for the temperature
        info = {
            'cost'           : 0,
            'extreme_penalty': 0,
        }
        for homie in self.homies:
            info[homie] = {}
            info[homie]["room"] = homie.current_room
            info[homie]["dt"] = self.current_dt
            info[homie]["temperature"] = self.temperatures[
                self.rooms[homie.current_room][0]]
            info[homie]["comfort"] = homie.get_preferred_temperature(
                self.current_dt)
            if not info[homie]["comfort"][0] <=\
                   info[homie]["temperature"] <=\
                   info[homie]["comfort"][1]:
                reward -= min(
                    abs(info[homie]["temperature"] - info[homie]["comfort"][0]),
                    abs(info[homie]["temperature"] - info[homie]["comfort"][1])
                ) ** 2
            
            homie.step(timestamp=self.current_dt)
        
        info['comfort_penalty'] = reward / self.homie_reward_scaler
        for room, cells in self.rooms.items():
            info[f'{room}_temperature'] = self.temperatures[cells[0]]
        
        # Adjust the temperature in the house wrt to the preferences of homies
        # action[action >= self.action_threshold] = 1
        # action[action < self.action_threshold] = 0
        self.change_temperature(self.action_matrices[action].copy())
        
        # Add costs for heating to reward
        if not self.action_names[action] == 'heat_Nothing':
            info['cost'] = 2
            reward -= 2
        
        done = self.step_count >= self.max_steps
        if self.temperatures.max() > 40:
            reward -= 100
            info['extreme_penalty'] = 100
        
        obs = self.gen_obs()
        
        # Remove the agent from the observation
        obs['image'][:, :, 0][obs['image'][:, :, 0] == 10] = 1
        
        # Logging
        if self.step_count % 100 == 0:
            colors = {
                'Kitchen'   : 'yellow',
                'Bathroom'  : 'magenta',
                'Bedroom'   : 'green',
                'LivingRoom': 'blue',
                'Outside'   : 'grey'
            }
            x = info[homie]
            del info[homie]
            print(termcolor.colored(
                text={**info, **x, 'action': self.action_names[action]},
                color=colors[x['room']]
            ))
            print(self.temperatures)

        return obs, reward, done, info
