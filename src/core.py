import numpy as np

from src.precinct import Precinct

class Core:

    def __init__(self,
                 precincts: list[Precinct]):
        
        self.precincts = precincts
        self.id_to_precinct = {p.precinct_id: i
                               for i, p in enumerate(self.precincts)}

        self.pop_vector = self.getForcingVector(data='population')
        self.laplacian = self.getLaplacian()
        self.core = self.getCore()

    def getForcingVector(self, data: str) -> np.ndarray:
    
        """
        type parameter may be:
            -> 'equal' for equal weights, 1.0
            -> 'population' for precinct populations
            -> 'republican' for GOP vote total
            -> 'democrat' for DEM vote total
            -> 'voting_pop' for sum of GOP and DEM votes
            -> 'vote' for GOP vote as % of total vote
        """

        # equal weights as default
        column = [[1.0] for p in self.precincts]

        if data == 'population':
            column = [[float(p.population)] for p in self.precincts]

        elif data == 'republican':
            column = [[float(p.gop_vote)] for p in self.precincts]

        elif data == 'democrat':
            column = [[float(p.dem_vote)] for p in self.precincts]

        elif data == 'voting_pop':
            column = [[float(p.gop_vote + p.dem_vote)] for p in self.precincts]

        elif data == 'vote':
            column = [[float(p.gop_vote / (p.gop_vote + p.dem_vote))] for p in self.precincts]

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