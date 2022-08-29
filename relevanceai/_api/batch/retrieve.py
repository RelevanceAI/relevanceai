# -*- coding: utf-8 -*-
"""Batch Retrieve"""

import math
import traceback

from typing import List, Optional

from relevanceai._api.batch.chunk import Chunker
from relevanceai._api.endpoints.api_client import APIEndpointsClient

from relevanceai.utils.cache import lru_cache
from relevanceai.utils.progress_bar import progress_bar

from relevanceai.constants.constants import MAX_CACHESIZE

# ADD SUPPORT FOR SAVING TO JSON


class BatchRetrieveClient(APIEndpointsClient, Chunker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_documents(
        self,
        dataset_id: str,
        number_of_documents: int = 20,
        filters: Optional[list] = None,
        cursor: str = None,
        chunksize: int = 1000,
        sort: Optional[list] = None,
        select_fields: Optional[list] = None,
        include_vector: bool = True,
        include_after_id: bool = False,
        include_cursor: bool = False,
        after_id: Optional[list] = None,
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
        chunksize: int
            Number of documents to retrieve per iteration
        include_vector: bool
            Include vectors in the search results
        sort: list
            Fields to sort by. For each field, sort by descending or ascending. If you are using descending by datetime, it will get the most recent ones.
        filters: list
            Query for filtering the search results
        """
        filters = [] if filters is None else filters
        sort = [] if sort is None else sort
        select_fields = [] if select_fields is None else select_fields

        if chunksize > number_of_documents:
            chunksize = number_of_documents

        resp = self.datasets.documents.get_where(
            dataset_id=dataset_id,
            select_fields=select_fields,
            include_vector=include_vector,
            page_size=chunksize,
            sort=sort,
            is_random=False,
            random_state=0,
            filters=filters,
            cursor=cursor,
            after_id=after_id,
        )
        data = resp["documents"]

        if number_of_documents > chunksize:
            after_id = resp["after_id"]
            _page = 0
            while resp:
                self.logger.debug(f"Paginating {_page} batch size {chunksize} ...")
                resp = self.datasets.documents.get_where(
                    dataset_id=dataset_id,
                    select_fields=select_fields,
                    include_vector=include_vector,
                    page_size=chunksize,
                    sort=sort,
                    is_random=False,
                    random_state=0,
                    filters=filters,
                    cursor=None,
                    after_id=after_id,
                )
                _data = resp["documents"]
                after_id = resp["after_id"]
                if (_data == []) or (after_id == []):
                    break
                data += _data
                if number_of_documents and (len(data) >= int(number_of_documents)):
                    break
                _page += 1
            data = data[:number_of_documents]

        if include_after_id:
            return {"documents": data, "after_id": resp["after_id"]}
        return data

    @lru_cache(maxsize=MAX_CACHESIZE)
    def _get_all_documents(
        self,
        dataset_id: str,
        chunksize: int = 1000,
        filters: Optional[List] = None,
        sort: Optional[List] = None,
        select_fields: Optional[List] = None,
        include_vector: bool = True,
        show_progress_bar: bool = True,
    ):
        """
        Retrieve all documents with filters. Filter is used to retrieve documents that match the conditions set in a filter query. This is used in advance search to filter the documents that are searched. For more details see documents.get_where.

        Example
        ---------

        >>> client = Client()
        >>> client.get_all_documents(dataset_id="sample_dataset_id"")

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        chunksize: list
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
        filters = [] if filters is None else filters
        sort = [] if sort is None else sort
        select_fields = [] if select_fields is None else select_fields

        # Initialise values
        length = 1
        after_id = None
        full_data = []

        # Find number of iterations
        number_of_documents = self.get_number_of_documents(
            dataset_id=dataset_id, filters=filters
        )
        iterations_required = math.ceil(number_of_documents / chunksize)

        # While there is still data to fetch, fetch it at the latest cursor

        for _ in progress_bar(
            range(iterations_required), show_progress_bar=show_progress_bar
        ):
            x = self.datasets.documents.get_where(
                dataset_id,
                filters=filters,
                after_id=after_id,
                page_size=chunksize,
                sort=sort,
                select_fields=select_fields,
                include_vector=include_vector,
            )
            try:
                length = len(x["documents"])
                after_id = x["after_id"]

                # Append fetched data to the full data
                if length > 0:
                    full_data += x["documents"]
            except Exception as e:
                traceback.print_exc()
                pass
        return full_data

    def get_number_of_documents(self, dataset_id, filters: Optional[List] = None):
        """
        Get number of documents in a dataset. Filter can be used to select documents that match the conditions set in a filter query. For more details see documents.get_where.

        Parameters
        ----------
        dataset_ids: list
            Unique names of datasets
        filters: list
            Filters to select documents
        """
        filters = [] if filters is None else filters

        return self.datasets.documents.get_where(
            dataset_id, page_size=1, filters=filters
        )["count"]
