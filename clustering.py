from diffusion import DiffusionMap

import numpy as np
from pathlib import Path
import pathlib
import pickle
from sklearn.cluster import KMeans

pathlib.PosixPath = pathlib.WindowsPath
path = Path('data') / 'eigenset97.pkl'

with open(path, 'rb') as f:
    diffusion: DiffusionMap = pickle.load(f)

eigenset = diffusion.eigenset
n = len(eigenset)

# number of diffusion coordinates to use
dim = 200

# number of clusters
k = 10

# skip eigenvector 0, usually the trivial/stationary eigenvector
dim = min(dim, len(eigenset) - 1)

# number of nodes/points being embedded
num_nodes = len(eigenset[0][1])

embedded = np.zeros((num_nodes, dim), dtype=float)

for coord in range(dim):
    embedded[:, coord] = np.array(eigenset[coord + 1][1], dtype=float)

kmeans = KMeans(
    n_clusters=k,
    random_state=0,
    n_init="auto"
)

labels = kmeans.fit_predict(embedded)
centers = kmeans.cluster_centers_

exit()

def plotCT(diffusion, eig_index: int):

    from pathlib import Path
    from collections import defaultdict
    import geopandas as gpd
    from bokeh.io import show
    from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, HoverTool
    from bokeh.palettes import Turbo256
    from bokeh.plotting import figure

    lam, vector = diffusion.eigenset[eig_index]

    path = Path(__file__).parent / "ct" / "shape" / "CT.shp"
    shape = gpd.read_file(path)

    weights = defaultdict(float)

    for node, value in zip(diffusion.ensemble.nodes, vector):
        precinct_name = node.precinct.precinct_name
        weights[precinct_name] += value

    shape["value"] = shape["NAME"].map(weights).fillna(0.0)

    xs, ys, names, vals = [], [], [], []

    for _, row in shape.iterrows():
        geometry = row.geometry

        polygons = (
            [geometry] if geometry.geom_type == "Polygon"
            else list(geometry.geoms) if geometry.geom_type == "MultiPolygon"
            else []
        )

        for polygon in polygons:
            xs.append(list(polygon.exterior.xy[0]))
            ys.append(list(polygon.exterior.xy[1]))
            names.append(row["NAME"])
            vals.append(row["value"])

    source = ColumnDataSource({
        "xs": xs,
        "ys": ys,
        "name": names,
        "value": vals,
    })

    max_abs = max(abs(v) for v in vals) or 1.0

    mapper = LinearColorMapper(
        palette=list(reversed(Turbo256)),
        low=-max_abs,
        high=max_abs,
    )

    fig = figure(
        title=f"lambda {eig_index} = {lam:.6f}",
        width=950,
        height=750,
        match_aspect=True,
        tools="pan,wheel_zoom,reset,save",
    )

    fig.patches(
        "xs",
        "ys",
        source=source,
        fill_color={"field": "value", "transform": mapper},
        line_color="black",
        line_width=0.25,
    )

    fig.add_tools(HoverTool(tooltips=[
        ("precinct", "@name"),
        ("value", "@value{0.000000}"),
    ]))

    fig.add_layout(ColorBar(color_mapper=mapper), "right")

    show(fig)

for i in [200]:
    plotCT(diffusion = diffusion,
        eig_index = i+1)


"""
k = 25
     1 ->   0.18053984642028809
     2 ->   3.0500988960266113
     4 ->  16.031352043151855
     8 ->  71.79584765434265
    10 -> 112.3741021156311
 
k = 50
     1 ->   0.20862865447998047
     2 ->   2.980881690979004
     4 ->  15.230807304382324
     8 ->  71.69842505455017
    10 -> 113.48672795295715

k = 100
     1 ->   0.26934313774108887
     2 ->   3.179410219192505
     4 ->  16.68631410598755
     8 ->  75.78099179267883
    10 -> 114.87385988235474

k = all using numpy
     1 ->   0.4290742874145508
     2 ->   4.801615476608276
     4 ->  25.79209280014038
     8 -> 133.15799927711487
    10 -> 227.17079734802246
 
"""