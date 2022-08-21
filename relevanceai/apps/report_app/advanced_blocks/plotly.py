import io
import numpy as np
from relevanceai.apps.report_app.blocks import ReportBlocks


class PlotlyReportBlock(ReportBlocks):
    """
    Adding a plotly block to Report
    """

    def plotly(
        self,
        fig,
        title: str = "",
        static: bool = False,
        width: int = None,
        height: int = None,
        add: bool = True,
        width_percentage: int = 100,
        options=None,
        **kwargs
    ):
        if options is None:
            options = {"displayLogo":False}
        try:
            import plotly
        except ImportError:
            raise ImportError(
                ".plotly requires plotly to be installed, install with 'pip install -U plotly'."
            )
        if static:
            try:
                import kaleido
            except ImportError:
                raise ImportError(
                    ".plotly 'static=True' requires kaleido to be installed, install with 'pip install -U kaleido'."
                )
            fig_image = fig.to_image(format="png", width=width, height=height)
            self.image(
                fig_image, title=title, width_percentage=width_percentage, add=add
            )
        else:
            layout = fig._layout
            if width:
                layout['width'] = width
            else:
                if "width" in layout and layout['width'] == np.inf:
                    layout['width'] = "auto"
            if height:
                layout['height'] = height
            else:
                if "height" in layout and layout['height'] == np.inf:
                    layout['height'] = "auto"
            block = {
                "type": "appBlock",
                # "attrs" : {"id": str(uuid.uuid4())},
                "content": [{
                    'attrs': {
                        'height': 'auto',
                        'data': fig._data,
                        'layout': layout,
                        'options': options,
                        'title': title,
                        'width': f'{width_percentage}%'
                    },
                    'type': 'plotlyChart'
                }]
            }
        if add:
            self.contents.append(block)
        return block


    def plotly_dendrogram(
        self,
        data,
        labels=None,
        hierarchy_method="single",
        metric="euclidean",
        orientation="right",
        color_threshold=0.75,
        **kwargs
    ):
        """
        Convenience function to plot a dendrogram.
        """
        from scipy.cluster import hierarchy

        linkage = hierarchy.linkage(data, method=hierarchy_method, metric=metric)
        try:
            import plotly.figure_factory as ff

            fig = ff.create_dendrogram(
                data,
                labels=labels,
                orientation=orientation,
                color_threshold=color_threshold,
                linkagefun=lambda x: linkage,
            )
            return self.plotly(fig, **kwargs)
        except ImportError:
            raise ImportError(
                "This requires plotly installed, install with `pip install -U plotly`"
            )
        except TypeError as e:
            raise TypeError(
                e
                + " This is a common error that can be fixed with `pip install pyyaml==5.4.1`"
            )