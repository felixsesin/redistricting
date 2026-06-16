import numpy as np

from src.district import District
from src.precinct import Precinct

class Node:

    # k-norm
    k = 2

    def __init__(self,
                 district: District,
                 precinct: Precinct):
        
        self.district = district
        self.precinct = precinct

        self.node_id = (district.district_id, precinct.precinct_id)

    def __str__(self) -> str:

        return (
            f"NODE(" +
            f"district={str(self.district)}\n" +
            f"     precinct={str(self.precinct)}\n" +
            f"     id={self.node_id})"
        )


    def __eq__(self, other: object) -> bool:

        if not isinstance(other, Node): return NotImplemented

        return self.node_id == other.node_id

    def canVerticallyDiffuseTo(self, other: object) -> bool:

        if not isinstance(other, Node): return NotImplemented

        return (self.district == other.district and
                self.precinct.doesNeighbour(other.precinct))
    
    def canHorizontallyDiffuseTo(self, other: object) -> bool:

        if not isinstance(other, Node): return NotImplemented

        return (self.district != other.district and
                self.precinct == other.precinct)
    
    def distanceTo(self, other: object) -> float:

        if not isinstance(other, Node): return NotImplemented

        core_1 = self.district.core
        avg, std = np.mean(core_1), np.std(core_1)
        core_1 = [(entry - avg) / std for entry in core_1]

        core_2 = other.district.core
        avg, std = np.mean(core_2), np.std(core_2)
        core_2 = [(entry - avg) / std for entry in core_2]

        precincts = set(self.district.precincts) | set(other.district.precincts)
        dist = 0.0

        for p in precincts:

            weight_1, weight_2 = 0.0, 0.0

            if p in self.district.precincts:
                j = self.district.precincts.index(p)
                weight_1 = float(core_1[j])


            if p in other.district.precincts:
                j = other.district.precincts.index(p)
                weight_2 = float(core_2[j])

            dist += abs(weight_2 - weight_1) ** self.k

        return np.power(dist, 1/self.k, dtype=float)