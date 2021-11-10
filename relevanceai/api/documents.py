from ..base import Base


class Documents(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def list(
        self,
        dataset_id: str,
        cursor: str = None,
        page_size: int = 20,
        sort: list = [],
        include_vector: bool = True,
        random_state: int = 0,
        output_format: str = "json",
        verbose: bool = True,
    ):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/list",
            method="GET",
            parameters={
                "cursor": cursor,
                "page_size": page_size,
                "sort": sort,
                "include_vector": include_vector,
                "random_state": random_state,
            },
            output_format=output_format,
            verbose=verbose,
        )

    def get(
        self,
        dataset_id: str,
        id: str,
        select_fields: list = [],
        cursor: str = None,
        page_size: int = 20,
        sort: list = [],
        include_vector: bool = True,
        output_format: str = "json",
        verbose: bool = True,
    ):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/get",
            parameters={
                "id": id,
                "select_fields": select_fields,
                "cursor": cursor,
                "page_size": page_size,
                "sort": sort,
                "include_vector": include_vector,
            },
            output_format=output_format,
            verbose=verbose,
        )

    def get_where(
        self,
        dataset_id: str,
        filters: list = [],
        cursor: str = None,
        page_size: int = 20,
        sort: list = [],
        select_fields: list = [],
        include_vector: bool = True,
        random_state: int = 0,
        is_random: bool = False,
        output_format: str = "json",
        verbose: bool = True,
    ):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/get_where",
            method="POST",
            parameters={
                "select_fields": select_fields,
                "cursor": cursor,
                "page_size": page_size,
                "sort": sort,
                "include_vector": include_vector,
                "filters": filters,
                "random_state": random_state,
                "is_random": is_random,
            },
            output_format=output_format,
            verbose=verbose,
        )

    def bulk_update(
        self,
        dataset_id: str,
        updates: list,
        output_format: str = "json",
        verbose: bool = True,
        return_documents: bool = False,
        retries=None,
    ):
        if return_documents is False:
            return self.make_http_request(
                endpoint=f"datasets/{dataset_id}/documents/bulk_update",
                method="POST",
                parameters={"updates": updates},
                output_format=output_format,
                verbose=verbose,
                retries=retries,
                base_url="https://ingest-api-dev-aueast.relevance.ai/latest/",
            )
        else:
            insert_response = self.make_http_request(
                endpoint=f"datasets/{dataset_id}/documents/bulk_update",
                method="POST",
                parameters={"updates": updates},
                output_format=False,
                verbose=verbose,
                retries=retries,
                base_url="https://ingest-api-dev-aueast.relevance.ai/latest/",
            )

            try:
                response_json = insert_response.json()
            except:
                response_json = None

            return {
                "response_json": response_json,
                "documents": updates,
                "status_code": insert_response.status_code,
            }

    def bulk_delete(
        self,
        dataset_id: str,
        ids: list = [],
        output_format: str = "json",
        verbose: bool = True,
    ):
        """Bulk delete."""
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/get_where",
            method="POST",
            parameters={"ids": ids},
            output_format=output_format,
            verbose=verbose,
        )

    def get_where_all(
        self,
        dataset_id: str,
        chunk_size: int = 10000,
        filters: list = [],
        sort: list = [],
        select_fields: list = [],
        include_vector: bool = True,
        random_state: int = 0,
        output_format: str = "json",
        verbose: bool = True,
    ):
        # Initialise values
        length = 1
        cursor = None
        full_data = []

        # While there is still data to fetch, fetch it at the latest cursor
        while length > 0:
            x = self.get_where(
                dataset_id,
                filters=filters,
                cursor=cursor,
                page_size=chunk_size,
                sort=sort,
                select_fields=select_fields,
                include_vector=include_vector,
                random_state=random_state,
                output_format=output_format,
                verbose=verbose,
            )
            length = len(x["documents"])
            cursor = x["cursor"]

            # Append fetched data to the full data
            if length > 0:
                [full_data.append(i) for i in x["documents"]]

        return full_data

    def get_number_of_documents(self, dataset_ids: list, list_of_filters=None):
        """
        dataset_ids: list of dataset_ids
        list_of_filters: list of list of filters corresponding to the same order of the dataset_ids
        """

        if list_of_filters is None:
            list_of_filters = [[] for _ in range(len(dataset_ids))]

        return {
            dataset_id: self._get_number_of_documents(dataset_id, filters)
            for dataset_id, filters in zip(dataset_ids, list_of_filters)
        }

    def _get_number_of_documents(self, dataset_id, filters=[]):
        """Certainty around the number of documents excluding chunks (until chunk documents is fixed)"""
        return self.get_where(dataset_id, page_size=1, filters=filters)["count"]
