import numpy as np

from src.district import District
from src.precinct import Precinct

class Node:

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
    
    def distanceTo(self,
                   other: object,
                   method: str,
                   mode: tuple | str) -> float:

        """
        Get distance between two nodes and specify:
            1) method = 'core'
                -> mode = (power, 'normal') or (power, mu, sig) or (mu_1, sig_1, mu_2, sig_2)

            2) method = 'hamming'
                -> mode = 'equal' or 'population' or 'relative' or
                          'republican' or 'democrat' or 'voting_pop' or 'vote'
        """

        if method == 'core':      return self.coreDistance(other, tuple(mode))
        elif method == 'hamming': return self.hammingDistance(other, mode) # type: ignore
        else:                              return NotImplemented

    def coreDistance(self,
                     other: object,
                     mode: tuple) -> float:

        if not isinstance(other, Node): return NotImplemented

        core_1 = self.district.core
        core_2 = other.district.core

        mu_1, sig_1 = np.mean(core_1), np.std(core_1)
        mu_2, sig_2 = np.mean(core_2), np.std(core_2)

        if len(mode) == 3:
            _, mu_1, sig_1 = mode
            _, mu_2, sig_2 = mode

        elif len(mode) == 5:
            _, mu_1, sig_1, mu_2, sig_2 = mode

        power = mode[0]

        core_1 = [(entry - mu_1) / sig_1 for entry in core_1]
        core_2 = [(entry - mu_2) / sig_2 for entry in core_2]

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

            dist += abs(weight_2 - weight_1) ** power # type: ignore

        return np.power(dist, 1/power, dtype=float) # type: ignore
    
    def hammingDistance(self,
                other: object,
                mode: str) -> float:
        
        if not isinstance(other, Node): return NotImplemented

        union = set(self.district.precincts) | set(other.district.precincts)
        diff  = set(self.district.precincts) ^ set(other.district.precincts)

        if mode == 'equal':
            return len(diff)

        elif mode == 'population':
            return sum(p.population for p in diff)

        elif mode == 'relative':
            pop, diff_pop = 0,0
            
            for p in union:
                pop += p.population
                if p in diff: diff_pop += p.population

            return diff_pop / pop

        elif mode == 'republican':
            return sum(p.gop_vote for p in diff)

        elif mode == 'democrat':
            return sum(p.dem_vote for p in diff)

        elif type == 'voting_pop':
            return sum([p.gop_vote + p.dem_vote for p in diff])

        elif mode == 'vote':
            return sum((p.gop_vote) / (p.dem_vote + p.gop_vote) for p in diff)

        else: return NotImplemented