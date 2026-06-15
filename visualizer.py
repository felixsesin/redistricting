from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

from diffusion import DiffusionMap


class Visualizer:

    def __init__(self, diffusion: DiffusionMap):
        
        self.diffusion = diffusion
        self.root = Path(__file__).parent
        self.img_dir = self.root / "imgs"
        self.shape_path = self.root / "ct" / "shape" / "CT.shp"

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
    ) -> None:
        import geopandas as gpd
        from bokeh.io import show
        from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, HoverTool
        from bokeh.palettes import Turbo256
        from bokeh.plotting import figure

        shape = gpd.read_file(self.shape_path)
        weights = self.precinctWeights(vector)

        shape["value"] = shape["NAME"].map(weights).fillna(0.0)

        xs, ys, names, vals = [], [], [], []

        for _, row in shape.iterrows():
            geometry = row.geometry

            if geometry.geom_type == "Polygon":
                polygons = [geometry]
            elif geometry.geom_type == "MultiPolygon":
                polygons = list(geometry.geoms)
            else:
                polygons = []

            for polygon in polygons:
                xs.append(list(polygon.exterior.xy[0]))
                ys.append(list(polygon.exterior.xy[1]))
                names.append(row["NAME"])
                vals.append(float(row["value"]))

        source = ColumnDataSource({
            "xs": xs,
            "ys": ys,
            "name": names,
            "value": vals,
        })

        max_abs = max((abs(v) for v in vals), default=1.0) or 1.0

        mapper = LinearColorMapper(
            palette=list(reversed(Turbo256)),
            low=-max_abs,
            high=max_abs,
        )

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