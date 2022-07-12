"""
Workflows Client
"""
from relevanceai.utils.base import _Base


class WorkflowsClient(_Base):
    def __init__(self, credentials):
        super().__init__(credentials)

    def trigger(self, params: dict, notebook_path: str, instance_type: str):
        """
        Trigger a workflow
        """
        return self.make_http_request(
            "/workflows/trigger",
            method="POST",
            parameters={
                "params": params,
                "notebook_path": notebook_path,
                "instance_type": instance_type,
            },
        )

    def list(self):
        return self.make_http_request(
            "/workflows/list",
            method="GET",
        )

    def get(self, workflow_id: str):
        """
        Returns

            {
                "job_status": "InProgress",
                "job_message": "string",
                "creation_time": "string",
                "notebook_path": "string",
                "params": { },
                "_id": "string",
                "metadata": { }
            }
        """
        return self.make_http_request(f"/workflows/{workflow_id}/get", method="POST")

    def metadata(self, workflow_id: str, metadata: dict):
        """
        Update metadata for a workflow run
        """
        return self.make_http_request(
            f"/workflows/{workflow_id}/metadata",
            method="POST",
            parameters={"metadata": metadata},
        )
