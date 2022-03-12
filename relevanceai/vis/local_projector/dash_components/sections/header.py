import dash
from dash import dcc
from dash import html


def build_header(app: dash.Dash) -> html.Div:
    """
    Builds the header of the app.
    """
    return html.Div(
        className="row header",
        id="app-header",
        style={"background-color": "#0F172A", "margin-bottom": "16px"},
        children=[
            # html.Div(
            #     [
            #         html.H3(
            #             "Vector Projector",
            #             className="header_title",
            #             id="app-title",
            #         )
            #     ],
            #     className="nine columns header_title_container",
            # ),
        ],
    )
