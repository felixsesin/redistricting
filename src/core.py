import numpy as np

from src.precinct import Precinct

class Core:

    def __init__(self,
                 precincts: list[Precinct]):
        
        self.precincts = precincts
        self.id_to_precinct = {p.precinct_id: i
                               for i, p in enumerate(self.precincts)}

        self.pop_vector = self.getPopVector()
        self.laplacian = self.getLaplacian()
        self.core = self.getCore()

    def getPopVector(self) -> np.ndarray:
    
        # precincts with equal weights
        # column = [[1.0] for p in self.precincts]

        # precincts weighted by population
        column = [[float(p.population)] for p in self.precincts]

        column.append([0.0])

        return np.array(column, dtype=float)
    
    def getLaplacian(self) -> np.ndarray:
        
        n = len(self.precincts)
        laplacian = np.zeros((n+1, n+1), dtype=float)

        for i, precinct in enumerate(self.precincts):

            degree = 0
            exterior = False

            for nbr in precinct.neighbours:

                if nbr in self.id_to_precinct:
                    degree += 1
                    j = self.id_to_precinct[nbr]
                    laplacian[i,j] = -1

                else:
                    exterior = True

            if exterior:
                degree += 1
                laplacian[i,n] = -1

            laplacian[i,i] = degree

        laplacian[n,n] = 1

        return laplacian
    
    def getCore(self) -> np.ndarray:

        return np.linalg.solve(self.laplacian, self.pop_vector).flatten()[:-1]