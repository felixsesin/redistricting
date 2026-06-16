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

diffusion = DiffusionMap(up_to=5,
                         alpha=0.03,
                         omega=0.5,
                         dim=50,
                         k_max=10)

visualize = Visualizer(diffusion)

for i in range(10):
    visualize.plotCluster(ind=i)