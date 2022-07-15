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
        width_percentage: int = 50,
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
