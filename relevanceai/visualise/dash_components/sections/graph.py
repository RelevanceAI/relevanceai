import dash
from dash import dcc
from dash import html
from relevanceai.visualise.dash_components.components.sections import Card


def build_graph(app: dash.Dash, data, layout) -> html.Div:
    """
    Builds the graph component of the layout.
    """
    import plotly.graph_objs as go

    return html.Div(
        className="six columns",
        children=[
            dcc.Graph(
                figure=go.Figure(data=data, layout=layout),
                id="graph-plot-tsne",
                style={"height": "100vh"},
                config={"displayModeBar": False},
            ),
        ],
    )
