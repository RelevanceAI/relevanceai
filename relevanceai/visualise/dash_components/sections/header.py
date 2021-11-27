import dash
from dash import dcc
from dash import html

def build_header(app: dash.Dash) -> html.Div:
    '''
    Builds the header of the app.
    '''
    return html.Div(
                    className="row header",
                    id="app-header",
                    style={"background-color": "#f9f9f9"},
                    children=[

                        html.Div(
                            [
                                html.H3(
                                    "Embedding Explorer",
                                    className="header_title",
                                    id="app-title",
                                )
                            ],
                            className="nine columns header_title_container",
                        ),
                    ],
                )
