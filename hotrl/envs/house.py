from typing import Tuple, List
from datetime import datetime, timedelta
import numpy as np

from gym_minigrid.envs.empty import MiniGridEnv
from gym_minigrid.minigrid import WorldObj, CELL_PIXELS, Grid

xy_coord = Tuple[int, int]
RoomType = ['Kitchen', 'Bathroom', 'Bedroom', 'LivingRoom', 'Outside']


class Homie(WorldObj):
    
    def __init__(self,
                 initial_room: RoomType = 'Kitchen',
                 color: int = -30,
                 ):
        super(Homie, self).__init__('ball', color)
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
        self.current_room = 'LivingRoom'
        self.cur_pos = self._place_within_the_room()
    
    def render(self, r):
        self._set_color(r)
        r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)
        
        
class HouseGrid(Grid):
    """
    A grid-world house with tenants and temperatures for each cell
    """
    pass
    

class House(MiniGridEnv):
    
    # TODO(Vlad): use multirooms instead with the name for each room
    # A mapping from coordinates to rooms
    cells_to_rooms = {
        (1, 1): 'LivingRoom',
        (1, 2): 'LivingRoom',
        (1, 3): 'LivingRoom',
    }
    rooms = {
        'Kitchen': [(2, 3), (3, 3)],
        'Bathroom': [(2, 1), (2, 2)],
        'Bedroom': [(3, 1), (3, 2)],
        'LivingRoom': [(1, 1), (1, 2), (1, 3)]
    }
    
    def __init__(
        self,
        size: int = 5,
        start_dt: datetime = datetime.now(),
        dt_delta: timedelta = timedelta(minutes=1),
        homies: List[Homie] = None,
    ):
        self.homies = homies
        self.current_dt = start_dt
        self.timedelta = dt_delta
        super().__init__(
            grid_size=5,
            max_steps=1000,
            see_through_walls=True,
        )
        
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Place tenants in the house
        for homie in self.homies:
            self.grid.set(*homie.cur_pos, v=homie)
        
        self.place_agent()

        self.mission = "it's getting hot in here"

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
        
        # Adjust the temperature in the house wrt to the preferences of homies
        # TODO (Andrei):
        
        if self.step_count >= self.max_steps:
            done = True
    
        obs = self.gen_obs()

        # Remove the agent from the observation
        self.grid
        obs['image'][:, :, 0][obs['image'][:, :, 0] == 10] = 1
    
        return obs, reward, done, {}
