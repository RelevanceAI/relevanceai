"""
All admin-related tasks.
"""
from typing import Optional

from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base


class AdminClient(_Base):
    def __init__(self, credentials: Credentials):
        self.project = credentials.project
        self.api_key = credentials.api_key

        super().__init__(credentials)

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
        source_project: Optional[str],
        source_api_key: Optional[str],
        project: str = None,
        api_key: str = None,
    ):
        """Copy a dataset from another user's projects into your project.

        Example
        -----------

        .. code-block::

            client = Client()
            client.copy_foreign_dataset(
                dataset_id="research",
                receiver_project="...",
                receiver_api_key="..."
            )

        Parameters
        -------------

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

    def send_dataset(
        self,
        dataset_id: str,
        receiver_project: str,
        receiver_api_key: str,
    ):
        """
        Send an individual a dataset.

        Example
        --------
        >>> client = Client()
        >>> client.admin.send_dataset(
            dataset_id="research",
            receiver_project="...",
            receiver_api_key="..."
        )

        Parameters
        -----------

        dataset_id: str
            The name of the dataset
        receiver_project: str
            The project name that will receive the dataset
        receiver_api_key: str
            The project API key that will receive the dataset

        """
        return self.make_http_request(
            "/admin/copy_foreign_dataset",
            method="POST",
            parameters={
                "project": receiver_project,
                "api_key": receiver_api_key,
                "dataset_id": dataset_id,
                "source_dataset_id": dataset_id,
                "source_project": self.project,
                "source_api_key": self.api_key,
                # "filters": filters
            },
        )

    def receive_dataset(
        self,
        dataset_id: str,
        sender_project: str,
        sender_api_key: str,
    ):
        """
        Receive an individual a dataset.

        Example
        --------
        >>> client = Client()
        >>> client.admin.receive_dataset(
            dataset_id="research",
            sender_project="...",
            sender_api_key="..."
        )

        Parameters
        -----------

        dataset_id: str
            The name of the dataset
        sender_project: str
            The project name that will send the dataset
        sender_api_key: str
            The project API key that will send the dataset

        """
        return self.make_http_request(
            "/admin/copy_foreign_dataset",
            method="POST",
            parameters={
                "dataset_id": dataset_id,
                "source_dataset_id": dataset_id,
                "project": self.project,
                "api_key": self.api_key,
                "source_project": sender_project,
                "source_api_key": sender_api_key,
                # "filters": filters
            },
        )

    def _ping(self):
        return self.make_http_request("/admin/ping", method="GET")
