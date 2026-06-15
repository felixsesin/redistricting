from clustering import Clustering
from ensemble import Ensemble
from horizontal import Horizontal
from vertical import Vertical

from scipy.sparse.linalg import eigs
import time
import pathlib
from pathlib import Path
import pickle

class DiffusionMap:

    def __init__(self,
                 up_to: int,
                 alpha: float,
                 omega: float,
                 dim: int,
                 k_max: int):
        
        start = time.time()

        self.up_to = up_to
        self.alpha = alpha
        self.omega = omega

        self.ensemble = Ensemble(up_to=up_to)
        print(f"GOT ENSEMBLE IN {time.time()-start}")
        start = time.time()

        self.vertical = Vertical(ensemble=self.ensemble).matrix
        print(f"GOT VERTICAL IN {time.time()-start}")
        start = time.time()

        self.horizontal = Horizontal(ensemble=self.ensemble,
                                     alpha=alpha).matrix
        print(f"GOT HORIZONTAL IN {time.time()-start}")
        start = time.time()

        self.diffusion = omega*self.vertical + (1-omega)*self.horizontal
        print(f"GOT DIFFUSION IN {time.time()-start}")
        start = time.time()

        self.eigenset = self.getEigenset()
        print(f"GOT EIGENSET IN {time.time()-start}")
        start = time.time()

        self.clusters = Clustering(dim, k_max, self.eigenset).clusters
        print(f"GOT CLUSTERS IN {time.time()-start}")
        start = time.time()


    @classmethod
    def load(cls, file: str):

        original = pathlib.PosixPath

        try:
            pathlib.PosixPath = pathlib.WindowsPath

            path = Path('data') / file

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
    
    def handlePlot(self,
                   fig,
                   ax,
                   plot_type: str = "NONE",
                   code: bool = False,
                   img: bool = False,
                   save: bool = False):

        import matplotlib.pyplot as plt

        if save:

            folder = Path(__file__).parent / 'imgs' / plot_type
            folder.mkdir(parents=True, exist_ok=True)

            filename = f"ct_small_{plot_type}_alpha_{self.alpha}_omega.png"
            path = folder / filename

            fig.savefig(path, dpi=300, bbox_inches='tight')

        if img:
            plt.show()

        if not code:
            plt.close(fig)

    def getSpectralDecay(self, code=False, img=False, save=False):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,6))

        lambdas = [abs(pair[0]) for pair in self.eigenset]
        indices = [i for i in range(len(lambdas))]

        ax.set_title(f"Spectral decay\nalpha={self.alpha}")
        ax.scatter(indices, lambdas)
        ax.grid(True)

        self.handlePlot(fig,ax,'spectral_decay',code,img,save)

        return None

    def getLambdaLambda(self, i: int, j: int, code=False, img=False, save=False):

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,10))

        lam1, vec1 = self.eigenset[i]
        lam2, vec2 = self.eigenset[j]

        ax.set_title(f"x: lambda {i} = {lam1:.6f}\ny: lambda {j} = {lam2:.6f}")
        ax.scatter(vec1, vec2)

        ax.axis('equal')
        ax.grid(True)

        self.handlePlot(fig,ax,'lambda_lambda',code,img,save)

        return None