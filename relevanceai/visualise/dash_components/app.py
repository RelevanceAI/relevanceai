import json
from traceback import print_tb

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output

from relevanceai.visualise.dash_components.sections.header import build_header
from relevanceai.visualise.dash_components.sections.display_panel import (
    build_display_panel,
)
from relevanceai.visualise.dash_components.sections.graph import build_graph
from relevanceai.visualise.dash_components.callbacks import (
    display_callbacks,
    neighbour_callbacks,
)
import warnings


def create_dash_graph(
    plot_data,
    layout,
    show_image: bool,
    docs: list,
    vector_label: str,
    vector_field: str,
    interactive: bool = True,
    style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
):

    from jupyter_dash import JupyterDash
    app = JupyterDash(__name__)

    def create_layout(app):
        """
        Create the layout of the Dash app.
        """
        return html.Div(
            className="row",
            style=style,
            children=[
                ## --- Header --- ##
                build_header(app),
                html.Div(id="my-output"),
                ## --- Body --- ##
                html.Div(
                    className="row background",
                    style={"padding": "0px"},
                    children=[
                        build_graph(app, plot_data, layout),
                        build_display_panel(app, show_image=show_image),
                    ],
                ),
            ],
        )

    app.layout = create_layout(app)
    # display_callbacks(app, show_image, docs, vector_label)
    if interactive:
        neighbour_callbacks(app, show_image, docs, vector_label, vector_field)
    app.run_server(mode="inline")


def create_dendrogram_tree(fig, services, dataset_id, field_name, node_label):
    from jupyter_dash import JupyterDash
    app = JupyterDash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(
                id="dendrogram",
                figure=fig,
            ),
            html.H1(id="body-div", style={"textAlign": "center"}),
        ]
    )

    @app.callback(Output("body-div", "children"), [Input("dendrogram", "hoverData")])
    def display_hover_data(hoverData):
        if hoverData is not None:
            x = hoverData["points"][0]["x"]
            y = hoverData["points"][0]["y"]
            index = [
                index
                for index, node in enumerate(fig.data)
                if node["x"][0] == x and node["y"][0] == y
            ][0]
            text = fig.data[index].text
            # if text is not None:
            #     mean_vec = json.loads(text)
            #     mean_vec = [value for value in mean_vec.values()]
            #     mvq = {"vector": mean_vec, "fields": field_name}
            # TODO:
            # Insert mean vec as centroids
            # list_closest_to_centers
            # search = services.search.vector(dataset_id, [mvq], page_size=1)
        return hoverData["points"][0][node_label]

    app.run_server(debug=True)
