from src.clustering import Clustering
from src.ensemble import Ensemble
from src.horizontal import Horizontal
from src.vertical import Vertical

from scipy.sparse.linalg import eigs
import time
import pathlib
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

class DiffusionMap:

    def __init__(self,
                 parameters: dict[str, int | float | str | tuple]):
        
        """
        Give HDM parameters like:

        parameters = {
            'up_to': 97,
            'omega': 0.5,
            'dim': 200,
            'k_max': 20,
            
            'method': 'core',
            'mode': (2, 'normal'),
            'data': 'population',

            'subtract': 0.0,
            'q_type': 'only_neighbours',

            'num_workers': 32
            'alpha': 0.3
        }
        """

        t_start = time.time()

        self.parameters = parameters

        self.up_to: int = parameters['up_to'] # type: ignore
        self.omega: float = parameters['omega'] # type: ignore
        self.dim: int = parameters['dim'] # type: ignore
        self.k_max: int = parameters['k_max'] # type: ignore

        self.ensemble = Ensemble(parameters = parameters)
        print(f"GOT ENSEMBLE IN {time.time()-t_start}")
        t_ensemble = time.time()
        self.tot_ensemble = t_ensemble - t_start

        self.vertical = Vertical(ensemble = self.ensemble,
                                 parameters = parameters).matrix
        print(f"GOT VERTICAL IN {time.time()-t_ensemble}")
        t_vertical = time.time()
        self.tot_vertical = t_vertical - t_ensemble

        self.horizontal = Horizontal(ensemble = self.ensemble,
                                     parameters = parameters).matrix
        print(f"GOT HORIZONTAL IN {time.time()-t_vertical}")
        t_horizontal = time.time()
        self.tot_horizontal = t_horizontal - t_vertical

        self.diffusion = self.omega*self.vertical + (1-self.omega)*self.horizontal
        print(f"GOT DIFFUSION IN {time.time()-t_horizontal}")
        t_diffusion = time.time()
        self.tot_diffusion = t_diffusion - t_horizontal

        self.eigenset = self.getEigenset()
        print(f"GOT EIGENSET IN {time.time()-t_diffusion}")
        t_eigenset = time.time()
        self.tot_eigenset = t_eigenset - t_diffusion

        #self.clusters = Clustering(self.dim, self.k_max, self.eigenset).clusters
        #print(f"GOT CLUSTERS IN {time.time()-t_eigenset}")
        #t_clusters = time.time()
        #self.tot_clusters = t_clusters - t_eigenset

        self.total_time = t_eigenset - t_start
        


    @classmethod
    def load(cls, file: str):

        original = pathlib.PosixPath

        try:
            pathlib.PosixPath = pathlib.WindowsPath

            path = DATA_DIR / file

            with open(path, 'rb') as f:
                obj = pickle.load(f)

        finally:
            pathlib.PosixPath = original

        return obj

    def getEigenset(self) -> list[tuple[float, list[float]]]:

        lambdas, eigenvectors = eigs(self.diffusion, # type: ignore
                                     k = 200,
                                     which = "LM")

        eigenlist = []

        for lam, vec in zip(lambdas, eigenvectors.T):

            if abs(lam.imag) > 1e-10:
                print(f"Warning: complex eigenvalue {lam}")

            eigenlist.append(
                (float(lam.real),
                vec.real.astype(float))
            )

        eigenlist.sort(
            key=lambda pair: abs(pair[0]),
            reverse=True
        )

        return [
            (lam, vec.tolist())
            for lam, vec in eigenlist
        ]
    
