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

