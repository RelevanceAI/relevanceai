import json
from collections import OrderedDict
from copy import deepcopy
from typing import Optional
import numpy as np

import scipy as scp
import scipy.spatial as scs
import scipy.cluster.hierarchy as sch

from plotly.graph_objs import graph_objs


class Dendrogram(object):
    """Refer to FigureFactory.create_dendrogram() for docstring."""

    def __init__(
        self,
        vectors,
        vector_field=None,
        orientation="bottom",
        labels=None,
        colorscale=None,
        width=np.inf,
        height=np.inf,
        xaxis="xaxis",
        yaxis="yaxis",
        distfun=None,
        linkagefun=lambda x: sch.linkage(x, "complete"),
        hovertext=None,
        color_threshold=None,
        number_of_nodes: Optional[int] = None,
        **kwargs
    ):
        self.vectors = vectors
        self.vector_field = vector_field

        try:
            self.node_label = kwargs["node_label"]
        except Exception:
            self.node_label = "_id"

        self.orientation = orientation
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}
        self.number_of_nodes = number_of_nodes

        if self.orientation in ["left", "bottom"]:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ["right", "bottom"]:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1

        if distfun is None:
            distfun = scs.distance.pdist

        (dd_traces, xvals, yvals, ordered_labels, leaves) = self.get_dendrogram_traces(
            vectors, colorscale, distfun, linkagefun, hovertext, color_threshold
        )

        self.labels = ordered_labels
        self.leaves = leaves
        yvals_flat = yvals.flatten()
        self.yvals_flat = yvals_flat
        xvals_flat = xvals.flatten()
        self.xvals_flat = xvals_flat

        self.zero_vals = []

        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0.0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])

        if len(self.zero_vals) > len(yvals) + 1:
            # If the length of zero_vals is larger than the length of yvals,
            # it means that there are wrong vals because of the identicial samples.
            # Three and more identicial samples will make the yvals of spliting
            # center into 0 and it will accidentally take it as leaves.
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(
                l_border, r_border + 1, int((r_border - l_border) / len(yvals))
            )
            # Regenerating the leaves pos from the self.zero_vals with equally intervals.
            self.zero_vals = [v for v in correct_leaves_pos]

        self.zero_vals.sort()
        self.layout = self.set_figure_layout(width, height)
        dd_traces = self.add_vertices(dd_traces)
        self.data = dd_traces

    def get(self):
        return graph_objs.Figure(data=self.data, layout=self.layout)

    def get_color_dict(self, colorscale):
        """
        Returns colorscale used for dendrogram tree clusters.

        :param (list) colorscale: Colors to use for the plot in rgb format.
        :rtype (dict): A dict of default colors mapped to the user colorscale.

        """

        # These are the color codes returned for dendrograms
        # We're replacing them with nicer colors
        # This list is the colors that can be used by dendrogram, which were
        # determined as the combination of the default above_threshold_color and
        # the default color palette (see scipy/cluster/hierarchy.py)
        d = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            # TODO: 'w' doesn't seem to be in the default color
            # palette in scipy/cluster/hierarchy.py
            "w": "white",
        }
        default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        if colorscale is None:
            rgb_colorscale = [
                "rgb(0,116,217)",  # blue
                "rgb(35,205,205)",  # cyan
                "rgb(61,153,112)",  # green
                "rgb(40,35,35)",  # black
                "rgb(133,20,75)",  # magenta
                "rgb(255,65,54)",  # red
                "rgb(255,255,255)",  # white
                "rgb(255,220,0)",  # yellow
            ]
        else:
            rgb_colorscale = colorscale

        for i in range(len(default_colors.keys())):
            k = list(default_colors.keys())[i]  # PY3 won't index keys
            if i < len(rgb_colorscale):
                default_colors[k] = rgb_colorscale[i]

        # add support for cyclic format colors as introduced in scipy===1.5.0
        # before this, the colors were named 'r', 'b', 'y' etc., now they are
        # named 'C0', 'C1', etc. To keep the colors consistent regardless of the
        # scipy version, we try as much as possible to map the new colors to the
        # old colors
        # this mapping was found by inpecting scipy/cluster/hierarchy.py (see
        # comment above).
        new_old_color_map = [
            ("C0", "b"),
            ("C1", "g"),
            ("C2", "r"),
            ("C3", "c"),
            ("C4", "m"),
            ("C5", "y"),
            ("C6", "k"),
            ("C7", "g"),
            ("C8", "r"),
            ("C9", "c"),
        ]
        for nc, oc in new_old_color_map:
            try:
                default_colors[nc] = default_colors[oc]
            except KeyError:
                # it could happen that the old color isn't found (if a custom
                # colorscale was specified), in this case we set it to an
                # arbitrary default.
                default_colors[n] = "rgb(0,116,217)"

        return default_colors

    def set_axis_layout(self, axis_key):
        """
        Sets and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.

        """
        axis_defaults = {
            "type": "linear",
            "ticks": "outside",
            "mirror": "allticks",
            "rangemode": "tozero",
            "showticklabels": True,
            "zeroline": False,
            "showgrid": False,
            "showline": True,
        }

        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ["left", "right"]:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]["tickvals"] = [
                zv * self.sign[axis_key] for zv in self.zero_vals
            ]
            self.layout[axis_key_labels]["ticktext"] = self.labels
            self.layout[axis_key_labels]["tickmode"] = "array"

        self.layout[axis_key].update(axis_defaults)

        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        """
        Sets and returns default layout object for dendrogram figure.

        """
        self.layout.update(
            {
                "showlegend": False,
                "autosize": False,
                "hovermode": "closest",
                "width": width,
                "height": height,
            }
        )

        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)

        return self.layout

    def get_dendrogram_traces(
        self, X, colorscale, distfun, linkagefun, hovertext, color_threshold
    ):
        """
        Calculates all the elements needed for plotting a dendrogram.

        :param (ndarray) X: Matrix of observations as array of arrays
        :param (list) colorscale: Color scale for dendrogram tree clusters
        :param (function) distfun: Function to compute the pairwise distance
                                   from the observations
        :param (function) linkagefun: Function to compute the linkage matrix
                                      from the pairwise distances
        :param (list) hovertext: List of hovertext for constituent traces of dendrogram
        :rtype (tuple): Contains all the traces in the following order:
            (a) trace_list: List of Plotly trace objects for dendrogram tree
            (b) icoord: All X points of the dendrogram tree as array of arrays
                with length 4
            (c) dcoord: All Y points of the dendrogram tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            (e) P['leaves']: left-to-right traversal of the leaves

        """
        d = distfun(X)
        Z = linkagefun(d)
        P = sch.dendrogram(
            Z,
            orientation=self.orientation,
            labels=self.labels,
            no_plot=True,
            color_threshold=color_threshold,
        )

        icoord = scp.array(P["icoord"])
        dcoord = scp.array(P["dcoord"])
        ordered_labels = scp.array(P["ivl"])
        color_list = scp.array(P["color_list"])
        colors = self.get_color_dict(colorscale)

        trace_list = []

        for i in range(len(icoord)):
            # xs and ys are arrays of 4 points that make up the '∩' shapes
            # of the dendrogram tree
            if self.orientation in ["top", "bottom"]:
                xs = icoord[i]
            else:
                xs = dcoord[i]

            if self.orientation in ["top", "bottom"]:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                marker=dict(color=colors[color_key]),
                text=hovertext_label,
                hoverinfo="text",
            )

            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ""

            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ""

            trace["xaxis"] = "x" + x_index
            trace["yaxis"] = "y" + y_index

            trace_list.append(trace)

        if self.number_of_nodes is None:
            return trace_list, icoord, dcoord, ordered_labels, P["leaves"]

        # Return the number of nodes
        return trace_list[:self.number_of_nodes], icoord[:self.number_of_nodes], dcoord[:self.number_of_nodes], ordered_labels[:self.number_of_nodes], P["leaves"]

    def add_vertices(self, dd_traces):
        vertices = []

        base = {
            "type": "scatter",
            "x": 0,
            "y": 0,
            "mode": "markers",
            "marker": {"color": "rgb(0, 0, 0)"},
            "text": None,
            "hoverinfo": "skip",
            "xaxis": "x",
            "yaxis": "y",
        }
        for trace in dd_traces:
            x = (trace["x"][1] + trace["x"][2]) / 2
            y = (trace["y"][1] + trace["y"][2]) / 2
            vertex = deepcopy(base)
            vertex["x"] = np.array(x)
            vertex["y"] = np.array(y)

            children = self.get_children(trace, dd_traces)
            children = [
                child for child in children if child["y"][0] == 0 or child["y"][3] == 0
            ]
            if not children:
                children = [trace]

            indices = self.get_indices(children)
            vectors = self.get_vectors(indices)
            mean_vec = np.mean(vectors, 0).tolist()

            vertex["text"] = json.dumps(
                {index: mean_vec[index] for index in range(len(mean_vec))}
            )
            vertices.append(vertex)

        return dd_traces + vertices

    def get_children(self, trace, dd_traces):
        children = [
            t
            for t in dd_traces
            if (
                (t["x"][1] < trace["x"][1] and trace["x"][1] < t["x"][2])
                or (t["x"][1] < trace["x"][2] and trace["x"][2] < t["x"][2])
            )
            and t["y"][1] < trace["y"][1]
        ]
        sub_children = []
        if children:
            for child in children:
                sub_children += self.get_children(child, dd_traces)

        return children + sub_children

    def get_indices(self, children):
        indices = []
        for child in children:
            x1 = child["x"][1]
            x2 = child["x"][2]
            y1 = child["y"][0]
            y2 = child["y"][3]

            if y1 == 0:
                indices.append(int(x1 / 10))
            if y2 == 0:
                indices.append(int(x2 / 10))
        return indices

    def get_vectors(self, indices):
        return np.array([self.vectors[index] for index in indices])
