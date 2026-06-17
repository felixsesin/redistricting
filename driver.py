'''

from diffusion import DiffusionMap

import pathlib
from pathlib import Path
import pickle

pathlib.PosixPath = pathlib.WindowsPath

"""
diffusion = DiffusionMap(up_to = 97,
                         alpha = 0.03,
                         omega = 0.5)
"""

path = Path('data') / 'eigenset97.pkl'

with open(path, 'rb') as f:
    diffusion: DiffusionMap = pickle.load(f)

'''

from src.diffusion import DiffusionMap
from src.visualizer import Visualizer

parameters = {
    'up_to': 3,
    'omega': 0.5,
    'dim': 200,
    'k_max': 20,

    'method': 'core',
    'mode': (2, 'normal'),
    'data': 'population',

    'subtract': 0.0,
    'q_type': 'only_neighbours',

    'alpha': 0.3
}

diffusion = DiffusionMap(parameters)

visualize = Visualizer(diffusion)

for i in range(10):
    visualize.plotEigenplan(ind=i)