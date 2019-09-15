import numpy as np

class SimpleModel:
    def __init__(self,
                 RSI: float = 4.2,
                 wall_height: float = 3.0,
                 wall_width: float = 10,
                 air_capacity: float = 1005,
                 air_density: float = 1.204,
                 timestep: int = 60,
                 round: int = 2,
                 ):
        """
        source for RSI:
        https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/energy/pdf/housing/Keeping-the-Heat-In_e.pdf
        (p15, Table 2-1)

        Good explanation of RSI:
        https://dothemath.ucsd.edu/2012/11/this-thermal-house/
        """
        wall_area = wall_width * wall_height
        self.A_over_RSI = wall_area / RSI
        room_mass = wall_area * wall_width * air_density
        self.air_capacity = air_capacity * room_mass
        assert(timestep <= 60), "timestep should be <= 60s to better " \
                                "approximate heat transfer..."
        self.timestep = timestep
        self.round = round

    def step(self, temperatures: np.ndarray, heat: np.ndarray):
        dT_y = temperatures[:, 1:] - temperatures[:, :-1]
        dT_x = temperatures[1:, :] - temperatures[:-1, :]

        heat_flow_y = dT_y * self.A_over_RSI * self.timestep
        heat_flow_x = dT_x * self.A_over_RSI * self.timestep

        heat[:, 1:] -= heat_flow_y
        heat[1:, :] -= heat_flow_x
        heat[:, :-1] += heat_flow_y
        heat[:-1, :] += heat_flow_x

        temperatures[1:-1, 1:-1] += heat[1:-1, 1:-1] / self.air_capacity
        heat *= 0
        return np.around(temperatures, self.round)

if __name__ == '__main__':
    house_shape = (10,5)
    T_start = 25
    T_out = 0

    wall_mask = np.ones(house_shape)
    wall_mask[1:-1, 1:-1] = 0

    temperatures = np.ones(house_shape) * T_start
    temperatures[wall_mask==1] = T_out

    heat = np.zeros(house_shape)

    sm = SimpleModel()
    for _ in range(1000):
        print(temperatures)
        temperatures = sm.step(temperatures, heat)


