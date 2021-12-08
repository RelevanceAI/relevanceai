import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from relevanceai.visualise.dash_components.components.sections import Card


def build_graph(app: dash.Dash, data, layout) -> html.Div:
    '''
    Builds the graph component of the layout.
    '''
    return html.Div(
        className='six columns',
        children=[
            Card(
                children=[
                    html.H3('Explore', className='mode_view',
                            id='mode-view', style={'text-align': 'center'}),
                    html.Div(id='div-plot-data-stats'),
                    html.Div(id='div-plot-cluster-stats'),
                ]
            ),
            dcc.Graph(figure=go.Figure(data=data, layout=layout), id='graph-plot-tsne',
                      style={'height': '100vh'},
                      config={'displayModeBar': False}),

        ],
    )
