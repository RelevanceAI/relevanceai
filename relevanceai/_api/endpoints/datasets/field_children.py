from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base


class FieldChildrenClient(_Base):
    """All dataset-related functions"""

    def __init__(self, credentials: Credentials):
        super().__init__(credentials)

    def list_field_children(self, _id: str, dataset_id: str, fieldchildren_id: str):
        """
        Returns the schema of a dataset. Refer to datasets.create for different field types available in a Relevance schema.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/field_children/{fieldchildren_id}/list",
            method="POST",
            parameters={"_id": _id},
        )

    def delete_field_children(self, dataset_id: str, fieldchildren_id: str):
        """
        Returns the schema of a dataset. Refer to datasets.create for different field types available in a Relevance schema.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/field_children/{fieldchildren_id}/delete",
            method="POST",
        )

    def update_field_children(self, _id: str, dataset_id: str, fieldchildren_id: str):
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/field_children/{fieldchildren_id}/update",
            method="POST",
            parameters={"_id": _id},
        )
