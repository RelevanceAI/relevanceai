"""Batch Retrieve"""

from typing import List
import math
from relevanceai.api.endpoints.client import APIClient
from relevanceai.api.batch.chunk import Chunker
from relevanceai.progress_bar import progress_bar

BYTE_TO_MB = 1024 * 1024
LIST_SIZE_MULTIPLIER = 3

# ADD SUPPORT FOR SAVING TO JSON


class BatchRetrieveClient(APIClient, Chunker):
    def get_documents(
        self,
        dataset_id: str,
        number_of_documents: int = 20,
        filters: list = [],
        cursor: str = None,
        batch_size: int = 1000,
        sort: list = [],
        select_fields: list = [],
        include_vector: bool = True,
    ):

        """
        Retrieve documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. \n
        If you are looking to combine your filters with multiple ORs, simply add the following inside the query {"strict":"must_or"}.
        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        number_of_documents: int
            Number of documents to retrieve
        select_fields: list
            Fields to include in the search results, empty array/list means all fields.
        cursor: string
            Cursor to paginate the document retrieval
        batch_size: int
            Number of documents to retrieve per iteration
        include_vector: bool
            Include vectors in the search results
        sort: list
            Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
        filters: list
            Query for filtering the search results
        """
        if batch_size > number_of_documents:
            batch_size = number_of_documents

        resp = self.datasets.documents.get_where(
            dataset_id=dataset_id,
            select_fields=select_fields,
            include_vector=include_vector,
            page_size=batch_size,
            sort=sort,
            is_random=False,
            random_state=0,
            filters=filters,
            cursor=cursor,
        )
        data = resp["documents"]

        if number_of_documents > batch_size:
            _cursor = resp["cursor"]
            _page = 0
            while resp:
                self.logger.debug(f"Paginating {_page} batch size {batch_size} ...")
                resp = self.datasets.documents.get_where(
                    dataset_id=dataset_id,
                    select_fields=select_fields,
                    include_vector=include_vector,
                    page_size=batch_size,
                    sort=sort,
                    is_random=False,
                    random_state=0,
                    filters=filters,
                    cursor=_cursor,
                )
                _data = resp["documents"]
                _cursor = resp["cursor"]
                if (_data == []) or (_cursor == []):
                    break
                data += _data
                if number_of_documents and (len(data) >= int(number_of_documents)):
                    break
                _page += 1
            data = data[:number_of_documents]

        return data

    def get_all_documents(
        self,
        dataset_id: str,
        chunk_size: int = 1000,
        filters: List = [],
        sort: List = [],
        select_fields: List = [],
        include_vector: bool = True,
        show_progress_bar: bool = True,
    ):
        """
        Retrieve all documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. For more details see documents.get_where.

        Example
        ---------

        >>> client = Client()
        >>> client.get_all_documents(dataset_id="sample_dataset"")

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        chunk_size : list
            Number of documents to retrieve per retrieval
        include_vector: bool
            Include vectors in the search results
        sort: list
            Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
        filters: list
            Query for filtering the search results
        select_fields : list
            Fields to include in the search results, empty array/list means all fields.
        """

        # Initialise values
        length = 1
        cursor = None
        full_data = []

        # Find number of iterations
        number_of_documents = self.get_number_of_documents(
            dataset_id=dataset_id, filters=filters
        )
        iterations_required = math.ceil(number_of_documents / chunk_size)

        # While there is still data to fetch, fetch it at the latest cursor

        for _ in progress_bar(
            range(iterations_required), show_progress_bar=show_progress_bar
        ):
            x = self.datasets.documents.get_where(
                dataset_id,
                filters=filters,
                cursor=cursor,
                page_size=chunk_size,
                sort=sort,
                select_fields=select_fields,
                include_vector=include_vector,
            )
            length = len(x["documents"])
            cursor = x["cursor"]

            # Append fetched data to the full data
            if length > 0:
                full_data += x["documents"]
        return full_data

    def get_number_of_documents(self, dataset_id, filters=[]):
        """
        Get number of documents in a dataset. Filter can be used to select documents that match the conditions set in a filter query. For more details see documents.get_where.

        Parameters
        ----------
        dataset_ids: list
            Unique names of datasets
        filters: list
            Filters to select documents
        """
        return self.datasets.documents.get_where(
            dataset_id, page_size=1, filters=filters
        )["count"]

    def get_vector_fields(self, dataset_id):
        """
        Returns list of valid vector fields in dataset
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        schema = self.datasets.schema(dataset_id)
        return [k for k in schema.keys() if k.endswith("_vector_")]
