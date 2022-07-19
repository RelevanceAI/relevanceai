import io
from relevanceai.apps.report_app.blocks import ReportBlocks


class PyplotReportBlock(ReportBlocks):
    """
    Adding a pyplot block to Report
    """

    def pyplot(
        self,
        fig,
        title: str = "",
        clear_fig: bool = True,
        add: bool = True,
        dpi: int = 100,
        width_percentage: int = 50,
        **kwargs
    ):
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
        except ImportError:
            raise ImportError(
                ".pyplot requires matplotlib to be installed, install with 'pip install -U matplotlib'."
            )
        fig_image = io.BytesIO()
        savefig_kwargs = {"dpi": dpi, "bbox_inches": "tight", "format": "png"}
        savefig_kwargs.update(kwargs)
        fig.savefig(fig_image, **savefig_kwargs)
        self.image(fig_image, title=title, width_percentage=width_percentage, add=add)
