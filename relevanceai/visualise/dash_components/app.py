import dash
from dash import html
from relevanceai.visualise.dash_components.sections.header import build_header
from relevanceai.visualise.dash_components.sections.display_panel import (
    build_display_panel,
)
from relevanceai.visualise.dash_components.sections.graph import build_graph
from relevanceai.visualise.dash_components.callbacks import (
    display_callbacks,
    neighbour_callbacks,
)
from jupyter_dash import JupyterDash


def create_dash_graph(
    plot_data,
    layout,
    show_image,
    docs,
    vector_label,
    vector_field,
    interactive: bool = True,
):

    app = JupyterDash(__name__)

    def create_layout(app: dash.Dash) -> html.Div:
        """
        Create the layout of the Dash app.
        """
        return html.Div(
            className="row",
            style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
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
