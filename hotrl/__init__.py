import os
from pathlib import Path

HOME_DIR = Path.home()
EXPERIMENTS_DIR = f'{HOME_DIR}/experiment_results'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))