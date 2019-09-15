from datetime import datetime, timedelta, time
from typing import Tuple, List

import numpy as np

from gym_minigrid.envs.empty import MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX, TEMPERATURES, COLORS
from gym_minigrid.minigrid import WorldObj, CELL_PIXELS, Grid, Floor

from hotrl.heat_transfer.model import SimpleModel

xy_coord = Tuple[int, int]
RoomType = ['Kitchen', 'Bathroom', 'Bedroom', 'LivingRoom', 'Outside']


class HeatingTile(Floor):
    
    def __init__(self, color='blue', temperature=20):
        super().__init__(color, temperature)


class Homie(WorldObj):
    
    def __init__(self,
                 initial_room: RoomType = 'Kitchen'):
        super(Homie, self).__init__('ball', color='blue')
        self.current_room = initial_room
        self.cur_pos = self._place_within_the_room()
    
    def _place_within_the_room(self) -> xy_coord:
        current_room = House.rooms[self.current_room]
        return current_room[np.random.randint(0, len(current_room) - 1)]
    
    def get_preferred_temperature(self, timestamp: datetime = None) -> int:
        """ Query homie for a preferred temperature at a current location,
        at a given time of the day or season"""
        
        if self.current_room == 'Kitchen':
            temp = 20
        elif self.current_room == 'Bathroom':
            temp = 22
        elif self.current_room == 'Bedroom':
            temp = 18
        elif self.current_room == 'LivingRoom':
            temp = 19
        else:
            raise ValueError('Undefined room type')
        
        return temp
    
    def step(self, timestamp: datetime) -> RoomType:
        date = dict(
            year=timestamp.year,
            month=timestamp.month,
            day=timestamp.day
        )
        prev_date = timestamp - timedelta(days=1)
        prev_date = dict(
            year=prev_date.year,
            month=prev_date.month,
            day=prev_date.day
        )
        
        sleep = datetime(**dict(**prev_date, hour=22))
        morning_bath = datetime(**dict(**date, hour=7))
        breakfast = datetime(**dict(**date, hour=7, minute=30))
        leave_for_work = datetime(**dict(**date, hour=8))
        dinner = datetime(**dict(**date, hour=18))
        study = datetime(**dict(**date, hour=19))
        evening_bath = datetime(**dict(**date, hour=21, minute=30))
        
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
        elif study <= evening_bath < evening_bath:
            self.current_room = 'LivingRoom'
        elif evening_bath <= evening_bath < sleep:
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
    """
    A grid-world house with tenants and temperatures for each cell
    """
    
    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
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
    # TODO(Vlad): use multirooms instead with the name for each room
    # A mapping from coordinates to rooms
    cells_to_rooms = {
        (1, 1): 'LivingRoom',
        (1, 2): 'LivingRoom',
        (1, 3): 'LivingRoom',
    }
    rooms = {
        'Kitchen'   : [(2, 3), (3, 3)],
        'Bathroom'  : [(2, 1), (2, 2)],
        'Bedroom'   : [(3, 1), (3, 2)],
        'LivingRoom': [(1, 1), (1, 2), (1, 3)],
        'Outside': [(0, 0)],
    }
    
    def __init__(
        self,
        temperatures: np.ndarray,
        size: int = 5,
        start_dt: datetime = datetime.now(),
        dt_delta: timedelta = timedelta(minutes=1),
        homies: List[Homie] = None,
    ):
        self.temperatures = temperatures
        self.homies = homies
        self.current_dt = start_dt
        self.timedelta = dt_delta
        self.model = SimpleModel()
        super().__init__(
            grid_size=5,
            max_steps=1000,
            see_through_walls=True,
        )
    
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
        done = False
        
        # Move each homie and determine their preference for the temperature
        rewards = dict()
        for homie in self.homies:
            homie.step(timestamp=self.current_dt)
            rewards[homie] = homie.get_preferred_temperature(self.current_dt)
        print(self.current_dt, rewards, homie.current_room)

        # Adjust the temperature in the house wrt to the preferences of homies
        self.change_temperature()
        
        if self.step_count >= self.max_steps:
            done = True
        
        obs = self.gen_obs()
        
        # Remove the agent from the observation
        obs['image'][:, :, 0][obs['image'][:, :, 0] == 10] = 1
        
        return obs, reward, done, {}
