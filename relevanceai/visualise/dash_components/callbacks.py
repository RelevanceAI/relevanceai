from dash import html
from PIL import Image
import numpy as np
import requests
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import numpy as np
from relevanceai.visualise.dash_components.utility.image_utility import resize, numpy_to_b64

MAX_SIZE = 200


def image_callbacks(app):
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
