from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import scipy.sparse as sp
from scipy.sparse import spmatrix

from src.ensemble import Ensemble

class Horizontal:

    def __init__(self,
                 ensemble: Ensemble,
                 parameters: dict[str, int | float | str | tuple]):
        
        self.num_workers: int = parameters['num_workers'] # type: ignore
        self.alpha: float = parameters['alpha'] # type: ignore
        self.beta: float = parameters['beta'] # type: ignore
        self.method: str = parameters['method'] # type: ignore
        self.mode: tuple | str = parameters['mode'] # type: ignore

        self.nodes = ensemble.nodes
        self.matrix = self.getHorizontalMatrix()

    def kernel(self, distance: float) -> float:

        # Gaussian kernel
        return float(np.exp(-self.alpha * distance * distance))

    def getHorizontalMatrix(self):

        n = len(self.nodes)

        num_workers = self.num_workers
        chunk_size = max(1, n // num_workers)

        tasks = []

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            tasks.append((
                self.nodes,
                self.alpha,
                self.method,
                self.mode,
                start,
                stop,
            ))

        all_rows = []
        all_cols = []
        all_data = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for rows, cols, data in executor.map(_horizontal_chunk, tasks):
                all_rows.extend(rows)
                all_cols.extend(cols)
                all_data.extend(data)

        H = sp.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(n, n),
            dtype=float,
        ).tocsr()

        row_sums = np.asarray(H.sum(axis=1)).ravel()
        beta = self.beta

        if beta != 0.0:
            scale = np.zeros_like(row_sums)
            mask = row_sums > 0
            scale[mask] = row_sums[mask] ** -beta
            
            D_beta = sp.diags(scale, format="csr")
            H = D_beta @ H @ D_beta

        row_sums = np.asarray(H.sum(axis=1)).ravel()

        inv = np.zeros_like(row_sums)
        mask = row_sums > 0
        inv[mask] = 1.0 / row_sums[mask]

        D_inv = sp.diags(inv, format="csr")

        return D_inv @ H


def _horizontal_chunk(args):
    nodes, alpha, method, mode, start, stop = args

    rows = []
    cols = []
    data = []

    n = len(nodes)

    for i in range(start, stop):
        node = nodes[i]

        for j in range(i + 1, n):
            neighbour = nodes[j]

            if not node.canHorizontallyDiffuseTo(neighbour):
                continue

            distance = node.distanceTo(
                neighbour,
                method=method,
                mode=mode,
            )

            weight = float(np.exp(-alpha * distance * distance))

            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([weight, weight])

    return rows, cols, data