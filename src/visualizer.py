from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from src.diffusion import DiffusionMap

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "imgs"
SHAPE_PATH = ROOT / "ct" / "shape" / "CT.shp"

class Visualizer:

    def __init__(self, diffusion: DiffusionMap):
        
        self.diffusion = diffusion
        self.root = ROOT
        self.img_dir = IMG_DIR
        self.shape_path = SHAPE_PATH

    def handleMatplotlib(self,
                         fig, plot_type: str,
                         code: bool = False, img: bool = False, save: bool = False
                         ) -> None:

        import matplotlib.pyplot as plt

        if save:
            self.img_dir.mkdir(parents=True, exist_ok=True)
            filename = (
                f"ct_small_{plot_type}"
                f"_alpha_{self.diffusion.alpha}"
                f"_omega_{self.diffusion.omega}.png"
            )
            fig.savefig(self.img_dir / filename, dpi=300, bbox_inches="tight")

        if img:
            plt.show()

        if not code:
            plt.close(fig)

    def spectralDecay(self,
                      code: bool = False, img: bool = False, save: bool = False
                      ) -> None:

        import matplotlib.pyplot as plt

        lambdas = [abs(lam) for lam, _ in self.diffusion.eigenset]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(
            f"Spectral decay\n"
            f"alpha={self.diffusion.alpha}, omega={self.diffusion.omega}"
        )
        ax.scatter(range(len(lambdas)), lambdas)
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("|lambda|")
        ax.grid(True)

        self.handleMatplotlib(fig, "spectral_decay", code, img, save)

    def lambdaLambda(self,
                     i: int, j: int,
                     code: bool = False, img: bool = False, save: bool = False
                     ) -> None:

        import matplotlib.pyplot as plt

        lam1, vec1 = self.diffusion.eigenset[i]
        lam2, vec2 = self.diffusion.eigenset[j]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(
            f"x: lambda {i} = {lam1:.6f}\n"
            f"y: lambda {j} = {lam2:.6f}"
        )
        ax.scatter(vec1, vec2)
        ax.set_xlabel(f"eigenvector {i}")
        ax.set_ylabel(f"eigenvector {j}")
        ax.axis("equal")
        ax.grid(True)

        self.handleMatplotlib(fig, f"lambda_{i}_lambda_{j}", code, img, save)

    def triLambda(self,
                  i: int, j: int, k: int,
                  code: bool = False, img: bool = False, save: bool = False
                  )-> None:

        import matplotlib.pyplot as plt

        lam1, vec1 = self.diffusion.eigenset[i]
        lam2, vec2 = self.diffusion.eigenset[j]
        lam3, vec3 = self.diffusion.eigenset[k]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_title(
            f"x: lambda {i} = {lam1:.6f}\n"
            f"y: lambda {j} = {lam2:.6f}\n"
            f"z: lambda {k} = {lam3:.6f}"
        )

        ax.scatter(vec1, vec2, vec3) # type: ignore

        ax.set_xlabel(f"eigenvector {i}")
        ax.set_ylabel(f"eigenvector {j}")
        ax.set_zlabel(f"eigenvector {k}")

        ax.grid(True)

        self.handleMatplotlib(
            fig,
            f"lambda_{i}_lambda_{j}_lambda_{k}",
            code,
            img,
            save,
        )

    def triLambdaPlotly(
        self,
        i: int,
        j: int,
        k: int,
        img: bool = True,
        save: bool = True,
    ) -> None:
        import plotly.graph_objects as go

        lam1, vec1 = self.diffusion.eigenset[i]
        lam2, vec2 = self.diffusion.eigenset[j]
        lam3, vec3 = self.diffusion.eigenset[k]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=vec1,
                    y=vec2,
                    z=vec3,
                    mode="markers",
                    marker=dict(size=3),
                )
            ]
        )

        fig.update_layout(
            title=(
                f"x: lambda {i} = {lam1:.6f}<br>"
                f"y: lambda {j} = {lam2:.6f}<br>"
                f"z: lambda {k} = {lam3:.6f}"
            ),
            scene=dict(
                xaxis_title=f"eigenvector {i}",
                yaxis_title=f"eigenvector {j}",
                zaxis_title=f"eigenvector {k}",
            ),
            width=950,
            height=750,
        )

        if save:
            self.img_dir.mkdir(parents=True, exist_ok=True)

            filename = (
                f"ct_small_lambda_{i}_lambda_{j}_lambda_{k}"
                f"_alpha_{self.diffusion.alpha}"
                f"_omega_{self.diffusion.omega}.html"
            )

            fig.write_html(self.img_dir / filename)

        if img:
            fig.show()

    def plotEigenplan(self, ind: int) -> None:
        
        lam, vector = self.diffusion.eigenset[ind]
        self.plotPrecincts(vector, ind=ind, lam=lam)

    def plotCluster(self, ind: int) -> None:
        
        from collections import Counter
        import numpy as np

        labels = self.diffusion.clusters[0]
        nodes = self.diffusion.ensemble.nodes
        k = len(labels)

        if ind < 0 or ind >= k:
            raise ValueError(
                f"cluster_index must be between 0 and {k-1}"
            )

        precinct_counts: Counter[str] = Counter()

        for node, label in zip(nodes, labels):
            if int(label) != ind:
                continue

            for precinct in node.district.precincts:
                precinct_counts[precinct.precinct_name] += 1

        # plotPrecincts() sums values from every node sharing a precinct.
        # Divide each counter among those occurrences so the final plotted
        # value equals the original counter.
        occurrences = Counter(
            node.precinct.precinct_name
            for node in nodes
        )

        vector = np.array([
            precinct_counts[node.precinct.precinct_name]
            / occurrences[node.precinct.precinct_name]
            for node in nodes
        ])

        self.plotPrecincts([float(x) for x in vector.flatten()], zero_as_missing=True)

    def precinctWeights(self, vector: Sequence[float]) -> dict[str, float]:
        
        weights = defaultdict(float)

        for node, value in zip(self.diffusion.ensemble.nodes, vector):
            precinct_name = node.precinct.precinct_name
            weights[precinct_name] += float(value)

        return dict(weights)

    def plotPrecincts(
        self,
        vector: Sequence[float],
        ind: int | None = None,
        lam: float | None = None,
        zero_as_missing: bool = False,
    ) -> None:
        import numpy as np
        import geopandas as gpd

        from bokeh.io import show
        from bokeh.models import (
            ColorBar,
            ColumnDataSource,
            HoverTool,
            LinearColorMapper,
        )
        from bokeh.palettes import Turbo256
        from bokeh.plotting import figure

        shape = gpd.read_file(self.shape_path)
        weights = self.precinctWeights(vector)

        # Unrepresented precincts become NaN rather than zero.
        shape["value"] = shape["NAME"].map(weights)

        if zero_as_missing:
            shape.loc[shape["value"] == 0, "value"] = np.nan

        xs, ys, names, vals = [], [], [], []

        for _, row in shape.iterrows():
            geometry = row.geometry

            if geometry.geom_type == "Polygon":
                polygons = [geometry]
            elif geometry.geom_type == "MultiPolygon":
                polygons = list(geometry.geoms)
            else:
                polygons = []

            value = row["value"]

            for polygon in polygons:
                xs.append(list(polygon.exterior.xy[0]))
                ys.append(list(polygon.exterior.xy[1]))
                names.append(row["NAME"])
                vals.append(float(value) if not np.isnan(value) else np.nan)

        finite_values = np.asarray(vals, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]

        if len(finite_values) == 0:
            low, high = -1.0, 1.0
        elif np.min(finite_values) >= 0:
            # Sequential scale for nonnegative counts.
            low = 0.0
            high = max(float(np.max(finite_values)), 1.0)
        else:
            # Symmetric scale for signed eigenvectors and core values.
            max_abs = max(float(np.max(np.abs(finite_values))), 1e-12)
            low, high = -max_abs, max_abs

        mapper = LinearColorMapper(
            palette=list(reversed(Turbo256)),
            low=low,
            high=high,
            nan_color="white",
        )

        source = ColumnDataSource({
            "xs": xs,
            "ys": ys,
            "name": names,
            "value": vals,
        })

        if ind is not None and lam is not None:
            title = f"lambda {ind} = {lam:.6f}"
        else:
            title = "Precinct vector"

        fig = figure(
            title=title,
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