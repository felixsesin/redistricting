





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

