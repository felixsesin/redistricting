




# from clustering

def plotCTT(diffusion, eig_index: int):

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



# from driver

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




def handlePlot(self,
                fig,
                ax,
                plot_type: str = "NONE",
                code: bool = False,
                img: bool = False,
                save: bool = False):

    import matplotlib.pyplot as plt

    if save:

        folder = Path(__file__).parent / 'imgs' / plot_type
        folder.mkdir(parents=True, exist_ok=True)

        filename = f"ct_small_{plot_type}_alpha_{self.alpha}_omega.png"
        path = folder / filename

        fig.savefig(path, dpi=300, bbox_inches='tight')

    if img:
        plt.show()

    if not code:
        plt.close(fig)

def getSpectralDecay(self, code=False, img=False, save=False):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))

    lambdas = [abs(pair[0]) for pair in self.eigenset]
    indices = [i for i in range(len(lambdas))]

    ax.set_title(f"Spectral decay\nalpha={self.alpha}")
    ax.scatter(indices, lambdas)
    ax.grid(True)

    self.handlePlot(fig,ax,'spectral_decay',code,img,save)

    return None

def getLambdaLambda(self, i: int, j: int, code=False, img=False, save=False):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,10))

    lam1, vec1 = self.eigenset[i]
    lam2, vec2 = self.eigenset[j]

    ax.set_title(f"x: lambda {i} = {lam1:.6f}\ny: lambda {j} = {lam2:.6f}")
    ax.scatter(vec1, vec2)

    ax.axis('equal')
    ax.grid(True)

    self.handlePlot(fig,ax,'lambda_lambda',code,img,save)

    return None