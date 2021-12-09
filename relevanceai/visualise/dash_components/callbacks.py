from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash import dcc

import plotly.express as px
from skimage import io
from plotly.subplots import make_subplots

from relevanceai.vector_tools.nearest_neighbours import NearestNeighbours
from doc_utils.doc_utils import DocUtils

doc_utils = DocUtils()

MAX_SIZE = 200

def display_callbacks(app, show_image):

    if show_image:
        @app.callback(Output('div-plot-click-image', 'children'), Input('graph-plot-tsne', 'clickData'))
        def display_image(clickData):
            try:
                image_url = clickData['points'][0]['customdata'][1]
            except TypeError:
                return None
            img = io.imread(image_url)
            fig = px.imshow(img)
            fig.update_yaxes(showgrid=False,showticklabels=False,linewidth=0)
            fig.update_xaxes(showgrid=False,showticklabels=False,linewidth=0)
            fig.update_layout(title="Current Selection")

            return dcc.Graph(
                            figure=fig,
                            style={'height': '50vh','width': '75', 'text-align': 'center'},
                            config={'displayModeBar': False},
                        )

    else:
        @app.callback(Output('div-plot-click-message', 'children'), Input('graph-plot-tsne', 'clickData'))
        def display_text(clickData):
            try: 
                return f"Current Selection: {clickData['points'][0]['customdata'][1]}"
            except TypeError:
                return None


def neighbour_callbacks(app, show_image, docs, vector_label, vector_field, distance_measure_mode = 'cosine'):
    def _get_neighbours(clickData):
        try:
            click_id = clickData['points'][0]['customdata'][0]
        except TypeError:
            return None  

        click_doc = [i for i in docs if i['_id'] == click_id][0]

        click_vec = click_doc[vector_field]
        nearest_neighbors = NearestNeighbours.get_nearest_neighbours(docs, click_vec, vector_field, distance_measure_mode)[1:11]
        nearest_neighbor_values = doc_utils.get_field_across_documents(vector_label, nearest_neighbors)
        nearest_neighbor_index = doc_utils.get_field_across_documents('nearest_neighbour_distance', nearest_neighbors)
        nearest_neighbor_index = [round(i, 2) for i in nearest_neighbor_index]

        if distance_measure_mode != 'cosine': 
            nearest_neighbor_index = nearest_neighbor_index[::-1]
            nearest_neighbor_values = nearest_neighbor_values[::-1]

        return {'nearest_neighbor_values': nearest_neighbor_values, 'nearest_neighbor_index':nearest_neighbor_index}

    if show_image:
        @app.callback(Output('div-plot-image-neighbours', 'children'), Input('graph-plot-tsne', 'clickData'))
        def image_neighbours(clickData):

            neighbour_info = _get_neighbours(clickData)
            if neighbour_info:

                fig = make_subplots(rows=5, cols=2, subplot_titles=neighbour_info['nearest_neighbor_index'])
                for n, image in enumerate(neighbour_info['nearest_neighbor_values']):
                    fig.add_trace(px.imshow(io.imread(image)).data[0], row=int(n/2)+1, col=n%2+1)
                    fig.update_yaxes(showgrid=False,showticklabels=False,linewidth=0)
                    fig.update_xaxes(showgrid=False,showticklabels=False,linewidth=0)

                fig.update_layout(title=f"Nearest Neighbours ({distance_measure_mode})")

                return dcc.Graph(
                                figure=fig,
                                style={'height': '125vh','width': '75vh', 'text-align': 'center',},
                                config={'displayModeBar': False},
                            )
            else:
                return None

    else:
        @app.callback(Output('div-plot-text-neighbours', 'children'), Input('graph-plot-tsne', 'clickData'))
        def text_neighbours(clickData):

            neighbour_info = _get_neighbours(clickData)

            if neighbour_info:
                trace = go.Bar(
                x=neighbour_info['nearest_neighbor_index'][::-1],
                y=neighbour_info['nearest_neighbor_values'][::-1],
                text = neighbour_info['nearest_neighbor_index'][::-1],
                orientation='h',
                marker=dict(color='rgb(24,84,255)')
                )
                layout = go.Layout(
                title=f"Nearest Neighbours ({distance_measure_mode})",
                xaxis={"visible": False},
                margin=go.layout.Margin(l=60, r=60, t=35, b=35),
                plot_bgcolor="#FFF",
                )
                fig = go.Figure(data=[trace], layout=layout)

                return dcc.Graph(
                                figure=fig,
                                style={'height': '125vh','width': '75vh', 'text-align': 'center',},
                                config={'displayModeBar': False},
                            )

            return None
    
