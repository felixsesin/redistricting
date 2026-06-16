import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from src.ensemble import Ensemble
from src.district import District

class Vertical:

    def __init__(self,
                 ensemble: Ensemble):
        
        self.districts = ensemble.districts
        self.matrix = self.getVerticalMatrix()

    def getVerticalMatrix(self) -> csr_matrix:

        """
        return vertical diffusion matrix for ensemble
        block diagonal with each block a transition matrix
        """

        blocks: list[csr_matrix] = []

        for district in self.districts:

            core = district.core
            total = sum(core)
            pi = core / total

            Q = self.getQ(type = 'only_neighbours',
                          district = district)

            block = self.metropolisHastings(pi = pi,
                                            Q  = Q)
            
            blocks.append(block)

        return sp.block_diag(blocks, format="csr") # type: ignore
    
    def metropolisHastings(self, pi: np.ndarray, Q: csr_matrix) -> csr_matrix:

        """
        return a transition matrix T whose stationary distribution is pi
        uses Metropolis-Hastings choice with initial transition matrix Q
        """

        Q = Q.tocoo() # type: ignore
        n = len(pi)

        rows = Q.row # type: ignore
        cols = Q.col # type: ignore
        q_ij = Q.data

        Q_csr = Q.tocsr()
        q_ji = np.array(Q_csr[cols, rows]).ravel()

        numerator = pi[cols] * q_ji
        denominator = pi[rows] * q_ij

        alpha = np.zeros_like(q_ij, dtype=float)

        mask = denominator > 0
        alpha[mask] = np.minimum(1.0, numerator[mask] / denominator[mask])

        offdiag_data = q_ij * alpha
        offdiag_mask = rows != cols

        P = sp.coo_matrix(
            (offdiag_data[offdiag_mask],
             (rows[offdiag_mask], cols[offdiag_mask])),
            shape=(n, n)
        ).tocsr()

        row_sums = np.asarray(P.sum(axis=1)).ravel()
        diagonal = 1.0 - row_sums

        P = P + sp.diags(diagonal, format="csr")

        return P
    
    def getQ(self, type: str, district: District) -> csr_matrix:

        """
        Choose initial matrix Q for Metropolis-Hastings:
            type = 'only_neighbours' or 'all_precincts'
        """

        if type == 'only_neighbours': return self.onlyNeighbours(district)

        elif type == 'all_precincts': return self.allPrecincts(len(district.precincts))

        else: return NotImplemented

    def onlyNeighbours(self, district: District) -> csr_matrix:

        """
        return the transition matrix for a simple random walk
        on the district, with equal probabilities P(i -> j)
        for all neighbours j of i
        """

        precincts = district.precincts
        n = len(precincts)

        index = {
            precinct.precinct_id: i
            for i, precinct in enumerate(precincts)
        }

        Q = sp.lil_matrix((n,n), dtype=float)

        for precinct in precincts:

            i = index[precinct.precinct_id]

            neighbours = [
                nbr
                for nbr in precinct.neighbours
                if nbr in index
            ]

            degree = len(neighbours)

            if degree == 0:
                Q[i, i] = 1.0
                continue

            prob = 1.0 / degree

            for nbr_id in neighbours:
                j = index[nbr_id]
                Q[i, j] = prob

        return Q.tocsr() # type: ignore
    
    def allPrecincts(self, n: int) -> csr_matrix:

        Q = sp.lil_matrix((n,n), dtype=float)
        Q[:, :] = 1 / (n - 1)

        for i in range(n): Q[i,i] = 0

        return Q.tocsr() # type: ignore