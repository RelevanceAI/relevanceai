from relevanceai.apps.report_app.sections import ReportSections


class ReportApp(ReportSections):
    def append_report(self, report):
        self.config["page-content"]["content"] = self.config["page-content"]["content"] + report.config["page-content"]["content"]

    def prepend_report(self, report):
        self.config["page-content"]["content"] = report.config["page-content"]["content"] + self.config["page-content"]["content"]
