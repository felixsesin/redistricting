import ct.network

from src.district import District
from src.node import Node
from src.precinct import Precinct

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CT_DIR = ROOT / "ct"
PLANS_DIR = CT_DIR / "plans"
SHAPE_DIR = CT_DIR / "shape"

class Ensemble:

    def __init__(self,
                 up_to: int):
        
        self.district_path = PLANS_DIR / 'small.jsonl'
        self.shape_path    = SHAPE_DIR / 'CT.shp'

        self.network = ct.network.info

        self.up_to = up_to
        self.precincts = self.getPrecincts()
        self.districts = self.getDistricts()
        self.nodes = self.getNodes()

    def getPrecinctByName(self, precinct: str) -> Precinct:

        for p in self.precincts:

            if p.precinct_name == precinct: return p

        raise ValueError('Precinct not found')

    def getPrecincts(self) -> list[Precinct]:

        return [
            Precinct(info, adjacency)
            for info, adjacency in
            zip(self.network['nodes'], self.network['adjacency'])
            ]
    
    def getDistricts(self) -> list[District]:

        districts = []

        with open(self.district_path) as f:
            for i, line in enumerate(f):

                if i < 3: continue
                if i == self.up_to + 3: break

                districting_plan = {k: [] for k in range(6)}
                districting = ast.literal_eval(line)['districting']

                for pair in districting:
                    for name, district in pair.items():

                        p = self.getPrecinctByName(name[2:-2])
                        districting_plan[district - 1].append(p)

                for d, precincts in districting_plan.items():

                    district = District(plan = i-3,
                                        district = d,
                                        precincts=precincts)
                    districts.append(district)

        return districts
    
    def getNodes(self) -> list[Node]:

        nodes = []

        for district in self.districts:
            for precinct in district.precincts:

                node = Node(district=district,
                            precinct=precinct)
                nodes.append(node)

        return nodes