import numpy as np
from sklearn.cluster import KMeans

class Clustering:

    def __init__(self,
                 dim: int,
                 k_max: int,
                 eigenset: list[tuple[float, list[float]]]):
        
        self.dim = dim
        self.k_max = k_max
        self.eigenset = eigenset

        self.embedded = self.getEmbedded()

        self.k = self.getK()
        self.clusters = self.getClusters()
        self.labels, self.centers = self.clusters

    def getEmbedded(self) -> np.ndarray:

        eigenset = self.eigenset
        n = len(eigenset)
        dim = min(self.dim, n-1)

        num_nodes = len(eigenset[0][1])
        embedded = np.zeros((num_nodes, dim), dtype=float)

        for coord in range(dim):
            embedded[:, coord] = np.array(eigenset[coord + 1][1], dtype=float)

        return embedded
    
    def getK(self) -> int:

        X = self.embedded
        n = len(X)

        k_max = min(self.k_max, n - 1)

        if k_max < 2:
            return 1

        ks = np.arange(1, k_max + 1)
        inertias = []

        for k in ks:
            kmeans = KMeans(
                n_clusters=k,
                random_state=0,
                n_init="auto"
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        inertias = np.array(inertias)

        # Normalize k and inertia so the geometry is scale-independent.
        x = (ks - ks.min()) / (ks.max() - ks.min())
        y = (inertias - inertias.min()) / (inertias.max() - inertias.min())

        points = np.column_stack([x, y])

        # Line from first point to last point.
        start = points[0]
        end = points[-1]
        line = end - start
        line_norm = np.linalg.norm(line)

        if line_norm == 0:
            return 1

        # Distance from each point to the line.
        distances = np.abs(
            np.cross(line, points - start) / line_norm
        )

        best_index = int(np.argmax(distances))
        return int(ks[best_index])

    def getClusters(self) -> tuple[np.ndarray, np.ndarray]:

        kmeans = KMeans(
            n_clusters   = self.k,
            random_state = 0,
            n_init       = "auto"
        )

        labels = kmeans.fit_predict(self.embedded)
        centers = kmeans.cluster_centers_

        return labels, centers


