import numpy as np

def step(temperatures: np.ndarray,
         heat_inputs: np.ndarray,
         conductivities: np.ndarray, # /thickness *area *time so units = J/deg
         capacities: np.ndarray, # *mass so units = J/deg
         outside: np.ndarray,
         ):
    # check all outer borders are constant temperatures
    assert (outside[:1, :] == 1).all()
    assert (outside[-2:, :] == 1).all()
    assert (outside[:, :1] == 1).all()
    assert (outside[:, -2:] == 1).all()

    # calculate temperature difference (dT) between two grid cells separated by a grid cell
    dT_y = temperatures[:, 2:] - temperatures[:, :-2]
    dT_x = temperatures[2:, :] - temperatures[:-2, :]
    """
    <--dT1-- 
       <--dT2--
          <--dT3--
             <--dT4--
    [ ][ ][ ][ ][ ][ ]
    +Q1
       +Q2
          -Q1
          +Q3
             -Q2
             +Q4
                -Q3         
                   -Q4      where Q_n = f(dT_n)
    """

    # calculate heat flows in and out of cells
    Q_y = conductivities[:,1:-1] * dT_y
    Q_x = conductivities[1:-1, :] * dT_x

    heat_inputs[:, :-2] += Q_y
    heat_inputs[:, 2:] -= Q_y
    heat_inputs[:-2, :] += Q_x
    heat_inputs[2:, :] -= Q_x

    # calculate temperature changes due to heat flows
    i = outside == 0
    temperatures[i] += heat_inputs[i]/capacities[i]

    return temperatures


if __name__ == '__main__':
    room_shape = (8,10)
    start_temp = 25
    outside_temp = 15
    num_steps = 10

    print("initializing temperatures...")
    temperatures = start_temp*np.ones(room_shape)
    print(temperatures)

    print("initializing outside mask")
    outside = np.zeros(room_shape)
    outside[:, :2] = 1
    outside[:, -2:] = 1
    outside[:2, :] = 1
    outside[-2:, :] = 1
    print(outside)

    print("updating outside temperatures...")
    temperatures[outside == 1] = outside_temp
    print(temperatures)

    print("initializing conductivities...")
    # https://www.engineeringtoolbox.com/thermal-conductivity-d_429.html
    timestep = 3600 # units: s
    air_thickness = 1 # units: m
    wall_thickness = 0.5 # units: m
    air_conductivity = 25/1000 # J/(m s)
    wall_conductivity = 0.5 # J/(m s)

    air_conductivity *= air_thickness
    air_conductivity *= timestep
    wall_conductivity *= wall_thickness
    wall_conductivity *= timestep

    conductivities = np.ones(room_shape) * air_conductivity
    conductivities[2,2:-2] = wall_conductivity
    conductivities[-3,2:-2] = wall_conductivity
    conductivities[2:-3,2] = wall_conductivity
    conductivities[2:-3, -3] = wall_conductivity
    print(conductivities)

    print("initializing capacities...")
    # https://www.engineeringtoolbox.com/specific-heat-capacity-d_391.html
    # https://www.engineeringtoolbox.com/air-density-specific-weight-d_600.html
    # https://www.engineeringtoolbox.com/bricks-density-d_1777.html
    height = 2.5 # units: m
    air_density = 1.204 # units: kg/m3
    wall_density = 1500  # units: m
    air_mass = air_thickness**2 * height * air_density  # units: kg
    wall_mass = wall_thickness * air_thickness * height * wall_density  # units: kg
    air_capacity = 1005  # J/(kg deg)
    wall_capacity = 850  # J/(kg deg)

    air_capacity *= air_mass
    wall_capacity *= wall_mass

    capacities = np.ones(room_shape) * air_capacity
    capacities[2, 2:-2] = wall_capacity
    capacities[-3, 2:-2] = wall_capacity
    capacities[2:-3, 2] = wall_capacity
    capacities[2:-3, -3] = wall_capacity
    print(capacities)

    print("initializing heat inputs...")
    heat_inputs = np.zeros(room_shape)

    np.set_printoptions(precision=0)
    for _ in range(num_steps):
        print(temperatures)
        temperatures = step(temperatures, heat_inputs, conductivities, capacities, outside)
        heat_inputs *= 0
