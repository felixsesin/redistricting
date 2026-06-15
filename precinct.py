from typing import TypedDict

class Info(TypedDict):
    id: int
    COUNTY: str
    NAME: str
    POP20: int
    G20PREDEM: int
    G20PREREP: int
    area: float
    border_length: float

class Adjacency(TypedDict):
    id: int
    length: float

class Precinct:

    def __init__(self,
                 info: Info,
                 adjacency: list[Adjacency]):

        """
        Neat wrapper for a precinct
        For CT, takes in any ct.network.info['nodes'][i]
        for i in range(695)
        """

        self.precinct_id   = info['id']
        self.county        = info['COUNTY']
        self.precinct_name = info['NAME']

        self.neighbours    = [neighbour['id'] for neighbour in adjacency]

        self.population    = info['POP20']
        
        self.dem_vote      = info['G20PREDEM']
        self.gop_vote      = info['G20PREREP']

        self.area          = info['area']
        self.border_length = info['border_length']

    def __str__(self) -> str:

        return (
            f"PLAN(" +
            f"county={self.county}, " +
            f"precinct={self.precinct_name}, " +
            f"id={self.precinct_id})"
        )

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, Precinct): return NotImplemented

        return self.precinct_id == other.precinct_id
    
    def doesNeighbour(self, other: object) -> bool:

        if not isinstance(other, Precinct): return NotImplemented

        return other.precinct_id in self.neighbours
    
    def __hash__(self) -> int:
        
        return hash(self.precinct_id)