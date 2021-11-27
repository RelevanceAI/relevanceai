# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import html
from relevanceai.visualise.dash_components.sections.header import build_header
from relevanceai.visualise.dash_components.sections.control_panel import build_control_panel
from relevanceai.visualise.dash_components.sections.display_panel import build_display_panel
from relevanceai.visualise.dash_components.sections.graph import build_graph
from relevanceai.visualise.dash_components.callbacks import image_callbacks
from jupyter_dash import JupyterDash


def create_dash_graph(data, layout):

    app = JupyterDash(__name__)

    def create_layout(app: dash.Dash) -> html.Div:
        '''
        Create the layout of the Dash app.
        '''
        return html.Div(
            className="row",
            style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
            children=[
                
                ## --- Header --- ##
                build_header(app),
                html.Div(id='my-output'),

                ## --- Body --- ##
                html.Div(
                    className="row background",
                    style={"padding": "0px"},
                    children=[
                        #build_control_panel(app),
                        build_graph(app, data, layout),
                        build_display_panel(app)
                    ], 
                ),
            ]
        )

    app.layout= create_layout(app)
    image_callbacks(app)
    app.run_server(mode='inline')


