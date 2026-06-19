from pathlib import Path
import pickle
import random

from src.diffusion import DiffusionMap
from src.visualizer import Visualizer

if __name__ == "__main__":

    path = Path(__file__).parent / 'data' / 'omega_trials' / 'omega_trials.pkl'

    dataset = {}

    with open(path, 'wb') as f:
        pickle.dump(dataset, f)

    omegas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]

    for omega in omegas:
         
        data = {
            'parameters': {'None': 'None'},
            'routine': 'None',
            'IV_omega': omega,
            'DV_eigenset': ['None']
        }

        parameters = {
            'up_to': 50,
            'omega': omega,
            'dim': 200,
            'k_max': 20,

            'method': 'core',
            'mode': (2, 'normal'),
            'data': 'population',

            'subtract': 0.0,
            'q_type': 'only_neighbours',

            'num_workers': 32,
            'alpha': 0.3,
            'beta': 0.0
        }

        # begin routine
        
        data['parameters'] = parameters
        diffusion = DiffusionMap(parameters)
        data['DV_eigenset'] = diffusion.eigenset
        
        # end routine

        data['routine'] = """

data['parameters'] = parameters
diffusion = DiffusionMap(parameters)
data['DV_eigenset'] = diffusion.eigenset

"""
            
            
        this_path = Path(__file__).parent / 'data' / 'omega_trials.pkl'

        with open(this_path, 'rb') as f:
            dataset = pickle.load(f)

        dataset[omega] = data

        with open(path, 'wb') as f:
            pickle.dump(dataset, f)        

        del this_path, data, parameters, diffusion

    
    
    





