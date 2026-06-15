import numpy as np
import scipy.sparse as sp
from scipy.sparse import spmatrix

from ensemble import Ensemble

class Horizontal:

    def __init__(self,
                 ensemble: Ensemble,
                 alpha: float):
        
        self.nodes = ensemble.nodes
        self.alpha = alpha

        self.matrix = self.getHorizontalMatrix()

    def kernel(self, distance: float) -> float:

        # Gaussian kernel
        return float(np.exp(-self.alpha * distance * distance))

    def getHorizontalMatrix(self) -> spmatrix:

        n = len(self.nodes)

        matrix = sp.lil_matrix((n,n), dtype=float)

        for i, node in enumerate(self.nodes):
            for j in range(i+1, n):

                neighbour = self.nodes[j]

                if not node.canHorizontallyDiffuseTo(neighbour): continue

                distance = node.distanceTo(neighbour)
                ker = self.kernel(distance)

                matrix[i,j] = ker
                matrix[j,i] = ker

        horizontal = matrix.tocsr()

        # row-normalize
        row_sums = np.asarray(horizontal.sum(axis=1)).ravel()
        inv_sums = np.zeros_like(row_sums)
        mask = row_sums > 0
        inv_sums[mask] = 1.0 / row_sums[mask]
        inv = sp.diags(inv_sums)


        return inv @ horizontal
    
