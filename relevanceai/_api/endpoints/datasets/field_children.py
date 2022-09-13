from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from typing import List


class FieldChildrenClient(_Base):
    """All dataset-related functions"""

    def __init__(self, credentials: Credentials):
        super().__init__(credentials)

    def list(self, dataset_id: str):
        """
        Returns the schema of a dataset. Refer to datasets.create for different field types available in a Relevance schema.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/field_children/list",
            method="POST",
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
    
    delete = delete_field_children 

    def update_field_children(
        self,
        dataset_id: str,
        fieldchildren_id: str,
        field: str,
        field_children: List,
        metadata: dict = None,
    ):
        if metadata is None:
            metadata = {}
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/field_children/{fieldchildren_id}/update",
            method="POST",
            parameters={
                "field": field,
                "field_children": field_children,
                "metadata": metadata,
            },
        )

    # make compatible with endpoint
    update = update_field_children
