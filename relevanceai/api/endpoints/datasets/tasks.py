"""Tasks Module
"""
import time

from relevanceai.base import _Base


class TasksClient(_Base):
    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    def create(self, dataset_id, task_name, task_parameters):
        """
        Tasks unlock the power of VecDb AI by adding a lot more new functionality with a flexible way of searching.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        task_name : string
            Name of task to complete
        task_parameters: dict
            Parameters of task to complete
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/create",
            method="POST",
            parameters={"task_name": task_name, **task_parameters},
        )

    def status(self, dataset_id: str, task_id: str):
        """
        Get status of a collection level job. Whether its starting, running, failed or finished.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        task_id : string
            Unique name of task
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/{task_id}/status", method="GET"
        )

    def list(self, dataset_id: str, show_active_only: bool = True):
        """
        List and get a history of all the jobs and its job_id, parameters, start time, etc.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        show_active_only : bool
            Whether to show active only
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/list",
            method="GET",
            parameters={
                "show_active_only": show_active_only,
            },
        )

    def _loop_status_until_finish(
        self,
        dataset_id: str,
        task_id: str,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):
        status = False

        while status not in ["Finished", "success"]:
            time.sleep(time_between_ping)
            try:
                status = self.status(dataset_id, task_id)["status"]
            except:
                self.logger.error(f"Status-check timed out: {task_id}")
                return task_id

            if verbose == True:
                self.logger.info(status)

        self.logger.success(f"Your task is {status}!")
        return

    def _check_status_until_finish(
        self,
        dataset_id: str,
        task_id: str,
        status_checker: bool = True,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):

        if status_checker == True:
            self.logger.info(f"Task_ID: {task_id}")
            self._loop_status_until_finish(
                dataset_id,
                task_id,
                verbose=verbose,
                time_between_ping=time_between_ping,
            )
            return

        else:
            self.logger.info(
                "To view the progress of your job, visit https://cloud.relevanceai.com/collections/dashboard/jobs"
            )
            return {"task_id": task_id}

    # Note: The following tasks are instantiated manually to accelerate
    # creation of certain popular tasks

    # Make decorator wrap for all task checkers

    def create_cluster_task(
        self,
        dataset_id,
        vector_field: str,
        n_clusters: int,
        alias: str = "default",
        refresh: bool = False,
        n_iter: int = 10,
        n_init: int = 5,
        status_checker: bool = True,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):

        """
        Start a task which creates clusters for a dataset based on a vector field
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        vector_field : string
            The field to cluster on.
        alias: string
            Alias is used to name a cluster
        n_clusters: int
            Number of clusters to be specified.
        n_iter: int
            Number of iterations in each run
        n_init: int
            Number of runs to run with different centroid seeds
        refresh: bool
            Whether to rerun task on the whole dataset or just the ones missing the output
        """

        task = self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/create",
            method="POST",
            parameters={
                "task_name": "Clusterer",
                "vector_field": vector_field,
                "n_clusters": n_clusters,
                "alias": alias,
                "refresh": refresh,
                "n_iter": n_iter,
                "n_init": n_init,
            },
        )

        output = self._check_status_until_finish(
            dataset_id,
            task["task_id"],
            status_checker=status_checker,
            verbose=verbose,
            time_between_ping=time_between_ping,
        )
        return output

    def create_numeric_encoder_task(
        self,
        dataset_id: str,
        fields: list,
        vector_name: str = "_vector_",
        status_checker: bool = True,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):
        """
        Within a collection encode the specified dictionary field in every document into vectors. \n
        For example: a dictionary that represents a person's characteristics visiting a store:
        >>> document 1 field: {"person_characteristics" : {"height":180, "age":40, "weight":70}}
        >>> document 2 field: {"person_characteristics" : {"age":32, "purchases":10, "visits": 24}}
        >>> -> <Encode the dictionaries to vectors> ->
        >>> | height | age | weight | purchases | visits |
        >>> |--------|-----|--------|-----------|--------|
        >>> | 180    | 40  | 70     | 0         | 0      |
        >>> | 0      | 32  | 0      | 10        | 24     |
        >>> document 1 dictionary vector: {"person_characteristics_vector_": [180, 40, 70, 0, 0]}
        >>> document 2 dictionary vector: {"person_characteristics_vector_": [0, 32, 0, 10, 24]}
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        fields : list
            The numeric fields to encode into vectors.
        vector_name: string
            The name of the vector field created
        """
        task = self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/create",
            method="POST",
            parameters={
                "task_name": "NumericEncoder",
                "fields": fields,
                "vector_name": vector_name,
            },
        )

        output = self._check_status_until_finish(
            dataset_id,
            task["task_id"],
            status_checker=status_checker,
            verbose=verbose,
            time_between_ping=time_between_ping,
        )
        return output

    def create_encode_categories_task(
        self,
        dataset_id: str,
        fields: list,
        status_checker: bool = True,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):
        """
        Within a collection encode the specified array field in every document into vectors. \n
        For example, array that represents a movie's categories:
        >>> document 1 array field: {"category" : ["sci-fi", "thriller", "comedy"]}
        >>> document 2 array field: {"category" : ["sci-fi", "romance", "drama"]}
        >>> -> <Encode the arrays to vectors> ->
        >>> | sci-fi | thriller | comedy | romance | drama |
        >>> |--------|----------|--------|---------|-------|
        >>> | 1      | 1        | 1      | 0       | 0     |
        >>> | 1      | 0        | 0      | 1       | 1     |
        >>> document 1 array vector: {"movie_categories_vector_": [1, 1, 1, 0, 0]}
        >>> document 2 array vector: {"movie_categories_vector_": [1, 0, 0, 1, 1]}

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        fields : list
            The numeric fields to encode into vectors.
        """
        task = self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/create",
            method="POST",
            parameters={"task_name": "CategoriesEncoder", "fields": fields},
        )

        output = self._check_status_until_finish(
            dataset_id,
            task["task_id"],
            status_checker=status_checker,
            verbose=verbose,
            time_between_ping=time_between_ping,
        )
        return output

    def create_encode_text_task(
        self,
        dataset_id: str,
        field: str,
        alias: str = "default",
        refresh: bool = False,
        status_checker: bool = True,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):
        """
        Start a task which encodes a text field
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        field : string
            The field to encode
        alias: string
            Alias used to name a vector field. Belongs in field_{alias}vector
        refresh: bool
            Whether to rerun task on the whole dataset or just the ones missing the output
        """

        task = self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/create",
            method="POST",
            parameters={
                "task_name": "TextEncoder",
                "dataset_id": dataset_id,
                "field": field,
                "alias": alias,
                "refresh": refresh,
            },
        )

        output = self._check_status_until_finish(
            dataset_id,
            task["task_id"],
            status_checker=status_checker,
            verbose=verbose,
            time_between_ping=time_between_ping,
        )
        return output

    def create_encode_textimage_task(
        self,
        dataset_id: str,
        field: str,
        alias: str = "default",
        refresh: bool = False,
        status_checker: bool = True,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):
        """
        Start a task which encodes a text field for image representation
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        field : string
            The field to encode
        alias: string
            Alias used to name a vector field. Belongs in field_{alias}vector
        refresh: bool
            Whether to rerun task on the whole dataset or just the ones missing the output
        """
        task = self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/create",
            method="POST",
            parameters={
                "task_name": "TextImageEncoder",
                "dataset_id": dataset_id,
                "field": field,
                "alias": alias,
                "refresh": refresh,
            },
        )

        output = self._check_status_until_finish(
            dataset_id,
            task["task_id"],
            status_checker=status_checker,
            verbose=verbose,
            time_between_ping=time_between_ping,
        )
        return output

    def create_encode_imagetext_task(
        self,
        dataset_id: str,
        field: str,
        alias: str = "default",
        refresh: bool = False,
        status_checker: bool = True,
        verbose: bool = True,
        time_between_ping: int = 10,
    ):
        """
        Start a task which encodes an image field for text representation
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        field : string
            The field to encode
        alias: string
            Alias used to name a vector field. Belongs in field_{alias}vector
        refresh: bool
            Whether to rerun task on the whole dataset or just the ones missing the output
        """
        task = self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/tasks/create",
            method="POST",
            parameters={
                "task_name": "ImageTextEncoder",
                "dataset_id": dataset_id,
                "field": field,
                "alias": alias,
                "refresh": refresh,
            },
        )

        output = self._check_status_until_finish(
            dataset_id,
            task["task_id"],
            status_checker=status_checker,
            verbose=verbose,
            time_between_ping=time_between_ping,
        )
        return output
