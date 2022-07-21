from relevanceai.apps.report_app.advanced_blocks.pyplot import PyplotReportBlock
from relevanceai.apps.report_app.advanced_blocks.plotly import PlotlyReportBlock


class ReportAdvancedBlocks(PyplotReportBlock, PlotlyReportBlock):
    def plot_by_method(self, plot, plot_method, title="", add=True):
        if plot_method == "plotly":
            self.plotly(plot, title=title, add=add)
        elif plot_method == "pyplot":
            self.pyplot(plot, title=title, add=add)

    # def plot_dendrogram()
