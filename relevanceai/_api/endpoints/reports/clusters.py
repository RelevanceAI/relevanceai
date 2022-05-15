from relevanceai.utils.base import _Base


class ClusterReportClient(_Base):
    def create(self, report: dict, name: str) -> dict:
        return self.make_http_request(
            "/reports/clusters/create",
            method="POST",
            parameters={"report": report, "name": name},
        )

    def get(self, report_id: str):
        return self.make_http_request(
            f"/reports/clusters/{report_id}/get",
            method="GET",
        )

    def update(self, report_id: str, report: dict, name: str):
        return self.make_http_request(
            f"/reports/clusters/{report_id}/update",
            method="POST",
            parameters={"report": report, "name": name},
        )

    def delete(self, report_id: str):
        return self.make_http_request(
            f"/reports/clusters/{report_id}/delete",
            method="POST",
        )

    def share(self):
        """Share all reports"""
        return self.make_http_request("/reports/clusters/share", method="POST")

    def private(self):
        return self.make_http_request("/reports/cluster/private", method="POST")

    def list(self):
        return self.make_http_request("/reports/clusters/list", method="GET")
