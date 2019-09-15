import numpy as np
import ray
from ray import tune
from ray.tune.registry import register_env

from hotrl import EXPERIMENTS_DIR
from hotrl.envs.house import House
from hotrl.envs.house_logger import HouseLogger
from hotrl.envs.wrappers import FullyObsWrapper
from hotrl.rllib_experiments.trainables import dqn_train

size = 4
inside_temp = 15.
outside_temp = 5.
temperatures = np.pad(
    np.full((size-2, size-2), fill_value=inside_temp),
    pad_width=[(1, 1), (1, 1)],
    mode='constant',
    constant_values=outside_temp
)
env_config = dict(
    size=size,
    homies_params=[{'initial_room': 'Bedroom'}],
    temperatures=temperatures
)

register_env("House", lambda config: FullyObsWrapper(House(**config)))
ray.init(
    # local_mode=True,
)

trials = tune.run(
    dqn_train,
    loggers=[HouseLogger],
    verbose=1,
    local_dir=EXPERIMENTS_DIR,
    config={
        "model"     : {
            # List of [out_channels, kernel, stride] for each filter
            "conv_filters": [
                [2, [4, 4], 1]
            ],
        },
        "env"       : "House",
        "env_config": env_config
    },
)
