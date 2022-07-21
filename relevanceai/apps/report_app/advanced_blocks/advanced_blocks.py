from relevanceai.apps.report_app.advanced_blocks.pyplot import PyplotReportBlock
from relevanceai.apps.report_app.advanced_blocks.plotly import PlotlyReportBlock
from relevanceai.apps.report_app.advanced_blocks.altair import AltairReportBlock


class ReportAdvancedBlocks(PyplotReportBlock, PlotlyReportBlock, AltairReportBlock):
    pass
