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
        width: int = None,
        height: int = None,
        clear_fig: bool = True,
        dpi: int = 150,
        add: bool = True,
        width_percentage: int = 100,
        **kwargs
    ):
        try:
            import matplotlib.pyplot as plt

            plt.ioff()
        except ImportError:
            raise ImportError(
                ".pyplot requires matplotlib to be installed, install with 'pip install -U matplotlib'."
            )
        if width or height:
            sizes = fig.get_size_inches()
            fig.set_size_inches(
                width / dpi if width else sizes[0], height / dpi if height else sizes[1]
            )
        fig_image = io.BytesIO()
        savefig_kwargs = {"dpi": dpi, "bbox_inches": "tight", "format": "png"}
        savefig_kwargs.update(kwargs)
        fig.savefig(fig_image, **savefig_kwargs)
        self.image(fig_image, title=title, width_percentage=width_percentage, add=add)
