"""
All admin-related tasks.
"""
from relevanceai.base import Base


class Admin(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def request_read_api_key(self, read_username: str):
        """Creates a read only key for your project. Make sure to save the api key somewhere safe. When doing a search the admin username should still be used.

        Parameters
        ----------
        read_username:
            Read-only project
        """
        return self.make_http_request(
            "/admin/request_read_api_key",
            method="POST",
            parameters={"read_username": read_username},
        )

    def copy_foreign_dataset(
        self,
        dataset_id: str,
        source_dataset_id: str,
        source_project: str,
        source_api_key: str,
        project: str=None,
        api_key: str=None,
    ):
        """Copy a dataset from another user's projects into your project.

        Parameters
        ----------
        dataset_id:
            The dataset to copy
        source_dataset_id:
            The original dataset
        source_project:
            The original project to copy from
        source_api_key:
            The original API key of the project
        project:
            The original project
        api_key:
            The original API key
        """
        return self.make_http_request(
            "/admin/copy_foreign_dataset",
            method="POST",
            parameters={
                "project": self.project if project is None else project,
                "api_key": self.api_key if api_key is None else api_key,
                "dataset_id": dataset_id,
                "source_dataset_id": source_dataset_id,
                "source_project": source_project,
                "source_api_key": source_api_key,
                # "filters": filters
            },
        )
