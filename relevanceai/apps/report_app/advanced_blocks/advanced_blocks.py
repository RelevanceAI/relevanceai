from relevanceai.apps.report_app.advanced_blocks.pyplot import PyplotReportBlock
from relevanceai.apps.report_app.advanced_blocks.plotly import PlotlyReportBlock


class ReportAdvancedBlocks(PyplotReportBlock, PlotlyReportBlock):
    def plot_by_method(self, plot, plot_method, add=True):
        if plot_method == "plotly":
            self.plotly(plot, add=add)
        elif plot_method == "pyplot":
            self.pyplot(plot, add=add)

    # def plot_dendrogram()
