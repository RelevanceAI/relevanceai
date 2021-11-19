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
        read_username : string
            Username for read only key
        """
        return self.make_http_request(
            "admin/request_read_api_key",
            method="POST",
            parameters={"read_username": read_username},
        )

    def copy_foreign_dataset(
        self,
        dataset_id: str,
        source_dataset_id: str,
        source_project: str,
        source_api_key: str,
        project: str = None,
        api_key: str = None,
        filters: list = []
    ):
        """Copy a dataset from another user's projects into your project. This is considered a project job
        Parameters
        ----------
        dataset_id : string
            Dataset name to copy into
        project: string
            Project name you want to copy the dataset into
        api_key: string
            Api key of the project you want to copy the dataset into
        source_dataset_id: string
            Dataset to copy frpm
        source_project: string
            Source project name of whom the dataset belongs to
        source_api_key: string
            Api key to access the source project name
        filters: string
            Query for filtering the dataset
        """
        return self.make_http_request(
            "admin/copy_foreign_dataset",
            method="POST",
            parameters={
                "project": self.project if project is None else project,
                "api_key": self.api_key if api_key is None else api_key,
                "dataset_id": dataset_id,
                "source_dataset_id": source_dataset_id,
                "source_project": source_project,
                "source_api_key": source_api_key,
                "filters": filters
            },
        )
