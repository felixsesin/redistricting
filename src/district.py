from src.core import Core
from src.precinct import Precinct

class District:

    def __init__(self,
                 plan: int,
                 district: int,
                 precincts: list[Precinct]):
        
        self.plan = plan
        self.district = district
        self.district_id = district + 6*plan

        self.precincts = precincts

        self.population = sum(p.population for p in precincts)
        self.dem_vote = sum(p.dem_vote for p in precincts)
        self.gop_vote = sum(p.gop_vote for p in precincts)
        self.area = sum(p.area for p in precincts)

        self.core = Core(self.precincts).core
    
    def __str__(self) -> str:

        return (
            f"DISTRICT(" +
            f"plan={self.plan}, " +
            f"district={self.district}, " +
            f"id={self.district_id})"
        )

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, District): return NotImplemented

        return self.district_id == other.district_id
    
    