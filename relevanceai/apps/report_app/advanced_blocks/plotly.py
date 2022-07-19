import io
from relevanceai.apps.report_app.blocks import ReportBlocks


class PlotlyReportBlock(ReportBlocks):
    """
    Adding a plotly block to Report
    """

    def plotly(
        self,
        fig,
        title: str = "",
        static: bool = True,
        width: int = 600,
        height: int = 300,
        add: bool = True,
        width_percentage: int = 100,
        **kwargs
    ):
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
                    ".plotly 'image=True' requires kaleido to be installed, install with 'pip install -U kaleido'."
                )
            fig_image = fig.to_image(format="png", width=width, height=height)
            self.image(
                fig_image, title=title, width_percentage=width_percentage, add=add
            )
        else:
            raise NotImplementedError

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
