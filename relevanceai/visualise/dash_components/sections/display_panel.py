import dash
from dash import dcc
from dash import html
from relevanceai.visualise.dash_components.components.sections import Card


def build_display_panel(app: dash.Dash, show_image: bool = True) -> html.Div:
    """
    Builds the display panel.
    """
    if show_image:
        children = Card(
            style={"width": "98vh", "textwd-align": "center"},
            children=[html.Div(id="div-plot-image-neighbours")],
        )
    else:
        children = Card(
            style={"width": "98vh", "textwd-align": "center"},
            children=[html.Div(id="div-plot-text-neighbours")],
        )
    return html.Div(
        className="three columns",
        id="display-panel",
        children=[children],
    )
