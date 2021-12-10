import dash
from dash import dcc
from dash import html
from relevanceai.visualise.dash_components.components.sections import Card
from relevanceai.visualise.dash_components.components.buttons import NamedSlider


def build_control_panel(app: dash.Dash) -> html.Div:
    """
    Builds the control panel for the embedding explorer.
    """
    return html.Div(
        className="three columns",
        children=[
            Card(
                [
                    NamedSlider(
                        name="Initial PCA Dimensions",
                        short="slider-pca-dimension",
                        min=2,
                        max=3,
                        step=None,
                        val=2,
                        marks={i: str(i) for i in [2, 3]},
                    ),
                ],
                className="card-style",
            ),
        ],
    )
