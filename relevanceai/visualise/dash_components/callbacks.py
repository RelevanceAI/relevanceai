from dash import html
from PIL import Image
import numpy as np
import requests
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import json
from dash import dcc

import numpy as np
from relevanceai.visualise.dash_components.utility.image_utility import resize, numpy_to_b64
from relevanceai.vector_tools.nearest_neighbours import NearestNeighbours
from doc_utils.doc_utils import DocUtils

doc_utils = DocUtils()

MAX_SIZE = 200


def image_callbacks(app):

    @app.callback(Output('div-plot-click-message', 'children'), Input('graph-plot-tsne', 'clickData'))
    def print_image_message(clickData):
        return json.dumps(clickData)

    @app.callback(Output('div-plot-click-image', 'children'), Input('graph-plot-tsne', 'clickData'))
    def show_image(clickData):
        try:
            image_url = clickData['points'][0]['customdata'][1]
        except TypeError:
            return None
        image_vector = np.array(Image.open(
            requests.get(image_url, stream=True).raw))
        image_vector = resize(image_vector, height=MAX_SIZE)
        image_b64 = numpy_to_b64(image_vector, scalar=False)
        return html.Img(
            src='data:image/png;base64, ' + image_b64,
            style={'height': '25vh', 'display': 'block', 'margin': 'auto'},
        )

def neighbour_callbacks(app, docs, field, vector_field, distance_measure_mode = 'cosine'):
    @app.callback(Output('div-plot-click-neighbours', 'children'), Input('graph-plot-tsne', 'clickData'))
    def show_neighbours(clickData):

        try:
            click_id = clickData['points'][0]['customdata'][0]
        except TypeError:
            return None  

        click_doc = [i for i in docs if i['_id'] == click_id][0]

        click_vec = click_doc[vector_field]
        click_value = click_doc[field]

        nearest_neighbors = NearestNeighbours.get_nearest_neighbours(docs, click_vec, vector_field, distance_measure_mode)[:10]
        nearest_neighbor_values = doc_utils.get_field_across_documents(field, nearest_neighbors)
        nearest_neighbor_index = doc_utils.get_field_across_documents('nearest_neighbour_distance', nearest_neighbors)

        trace = go.Bar(
        x=nearest_neighbor_values,
        y=nearest_neighbor_index,
        marker=dict(color='rgb(24,84,255)'),
        )

        layout = go.Layout(
        title=f"{click_value} Nearest Neighbours",
        xaxis=dict(title=f'{distance_measure_mode} Distance'),
        margin=go.layout.Margin(l=60, r=60, t=35, b=35),
        )
        fig = go.Figure(data=[trace], layout=layout)

        return dcc.Graph(
                        id='graph-bar-nearest-neighbors',
                        figure=fig,
                        style={'height': '25vh','width': '50vh', 'text-align': 'center',},
                        config={'displayModeBar': False},
                    )


