"""All Dataset related functions
"""
from typing import List, Optional

from relevanceai.client.helpers import Credentials
from relevanceai.utils.base import _Base
from relevanceai._api.endpoints.datasets.documents import DocumentsClient
from relevanceai._api.endpoints.datasets.monitor import MonitorClient
from relevanceai._api.endpoints.datasets.tasks import TasksClient
from relevanceai._api.endpoints.datasets.cluster import ClusterClient
from relevanceai._api.endpoints.datasets.field_children import FieldChildrenClient


class DatasetsClient(_Base):
    """All dataset-related functions"""

    def __init__(self, credentials: Credentials):
        self.tasks = TasksClient(credentials)
        self.documents = DocumentsClient(credentials)
        self.monitor = MonitorClient(credentials)
        self.cluster = ClusterClient(credentials)
        self.field_children = FieldChildrenClient(credentials)

        super().__init__(credentials)

    def schema(self, dataset_id: str):
        """
        Returns the schema of a dataset. Refer to datasets.create for different field types available in a Relevance schema.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/schema", method="GET"
        )

    def metadata(self, dataset_id: str):
        """
        Retreives metadata about a dataset. Notably description, data source, etc

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/metadata", method="GET"
        )

    def post_metadata(self, dataset_id: str, metadata: dict):
        """
        Edit and add metadata about a dataset. Notably description, data source, etc
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/metadata",
            method="POST",
            parameters={"dataset_id": dataset_id, "metadata": metadata},
        )

    def create(self, dataset_id: str, schema: Optional[dict] = None):
        """
        A dataset can store documents to be searched, retrieved, filtered and aggregated (similar to Collections in MongoDB, Tables in SQL, Indexes in ElasticSearch).
        A powerful and core feature of Relevance is that you can store both your metadata and vectors in the same document. When specifying the schema of a dataset and inserting your own vector use the suffix (ends with) "_vector_" for the field name, and specify the length of the vector in dataset_schema. \n

        For example:

        >>>    {
        >>>        "product_image_vector_": 1024,
        >>>        "product_text_description_vector_" : 128
        >>>    }

        These are the field types supported in our datasets: ["text", "numeric", "date", "dict", "chunks", "vector", "chunkvector"]. \n

        For example:

        >>>    {
        >>>        "product_text_description" : "text",
        >>>        "price" : "numeric",
        >>>        "created_date" : "date",
        >>>        "product_texts_chunk_": "chunks",
        >>>        "product_text_chunkvector_" : 1024
        >>>    }

        You don't have to specify the schema of every single field when creating a dataset, as Relevance will automatically detect the appropriate data type for each field (vectors will be automatically identified by its "_vector_" suffix). Infact you also don't always have to use this endpoint to create a dataset as /datasets/bulk_insert will infer and create the dataset and schema as you insert new documents. \n

        Note:

            - A dataset name/id can only contain undercase letters, dash, underscore and numbers.
            - "_id" is reserved as the key and id of a document.
            - Once a schema is set for a dataset it cannot be altered. If it has to be altered, utlise the copy dataset endpoint.

        For more information about vectors check out the 'Vectorizing' section, services.search.vector or out blog at https://relevance.ai/blog. For more information about chunks and chunk vectors check out datasets.search.chunk.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        schema : dict
            Schema for specifying the field that are vectors and its length

        """
        schema = {} if schema is None else schema

        return self.make_http_request(
            endpoint=f"/datasets/create",
            method="POST",
            parameters={"id": dataset_id, "schema": schema},
        )

    def list(self):
        """List all datasets in a project that you are authorized to read/write."""
        return self.make_http_request(endpoint="/datasets/list", method="GET")

    def list_all(
        self,
        include_schema: bool = True,
        include_stats: bool = True,
        include_metadata: bool = True,
        include_schema_stats: bool = False,
        include_vector_health: bool = False,
        include_active_jobs: bool = False,
        dataset_ids: Optional[list] = None,
        sort_by_created_at_date: bool = False,
        asc: bool = False,
        page_size: int = 20,
        page: int = 1,
    ):
        """
        Returns a page of datasets and in detail the dataset's associated information that you are authorized to read/write. The information includes:

        - Schema - Data schema of a dataset (same as dataset.schema).
        - Metadata - Metadata of a dataset (same as dataset.metadata).
        - Stats - Statistics of number of documents and size of a dataset (same as dataset.stats).
        - Vector_health - Number of zero vectors stored (same as dataset.health).
        - Schema_stats - Fields and number of documents missing/not missing for that field (same as dataset.stats).
        - Active_jobs - All active jobs/tasks on the dataset.

        Parameters
        ----------
        include_schema : bool
            Whether to return schema
        include_stats : bool
            Whether to return stats
        include_metadata : bool
            Whether to return metadata
        include_vector_health : bool
            Whether to return vector_health
        include_schema_stats : bool
            Whether to return schema_stats
        include_active_jobs : bool
            Whether to return active_jobs
        dataset_ids : list
            List of dataset IDs
        sort_by_created_at_date : bool
            Sort by created at date. By default shows the newest datasets. Set asc=False to get oldest dataset.
        asc : bool
            Whether to sort results by ascending or descending order
        page_size : int
            Size of each page of results
        page : int
            Page of the results
        """
        dataset_ids = [] if dataset_ids is None else dataset_ids

        return self.make_http_request(
            endpoint="/datasets/list",
            method="POST",
            parameters={
                "include_schema": include_schema,
                "include_stats": include_stats,
                "include_metadata": include_metadata,
                "include_schema_stats": include_schema_stats,
                "include_vector_health": include_vector_health,
                "include_active_jobs": include_active_jobs,
                "dataset_ids": dataset_ids,
                "sort_by_created_at_date": sort_by_created_at_date,
                "asc": asc,
                "page_size": page_size,
                "page": page,
            },
        )

    def facets(
        self,
        dataset_id,
        fields: Optional[list] = None,
        date_interval: str = "monthly",
        page_size: int = 5,
        page: int = 1,
        asc: bool = False,
    ):
        """
        Takes a high level aggregation of every field, return their unique values and frequencies. This is used to help create the filter bar for search.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        fields : list
            Fields to include in the facets, if [] then all
        date_interval : str
            Interval for date facets
        page_size : int
            Size of facet page
        page : int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order

        """
        fields = [] if fields in (None, [None]) else fields

        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/facets",
            method="POST",
            parameters={
                "fields": fields,
                "date_interval": date_interval,
                "page_size": page_size,
                "page": page,
                "asc": asc,
            },
        )

    def check_missing_ids(self, dataset_id, ids):

        """
        Look up in bulk if the ids exists in the dataset, returns all the missing one as a list.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        ids : list
            IDs of documents
        """

        # Check if dataset_id exists
        dataset_exists = dataset_id in self.list()["datasets"]

        if dataset_exists:
            return self.make_http_request(
                endpoint=f"/datasets/{dataset_id}/documents/get_missing",
                method="GET",
                parameters={"ids": ids},
            )

        else:
            print("Dataset does not exist")
            return

    def insert(
        self,
        dataset_id: str,
        document: dict,
        insert_date: bool = True,
        overwrite: bool = True,
        update_schema: bool = True,
    ):
        """
        Insert a single documents

        - When inserting the document you can optionally specify your own id for a document by using the field name "_id", if not specified a random id is assigned.
        - When inserting or specifying vectors in a document use the suffix (ends with) "_vector_" for the field name. e.g. "product_description_vector_".
        - When inserting or specifying chunks in a document the suffix (ends with) "_chunk_" for the field name. e.g. "products_chunk_".
        - When inserting or specifying chunk vectors in a document's chunks use the suffix (ends with) "_chunkvector_" for the field name. e.g. "products_chunk_.product_description_chunkvector_".

        Documentation can be found here: https://ingest-api-dev-aueast.relevance.ai/latest/documentation#operation/InsertEncode \n

        Try to keep each batch of documents to insert under 200mb to avoid the insert timing out. \n

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        document : dict
            Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
        insert_date : bool
            Whether to include insert date as a field 'insert_date_'.
        overwrite : bool
            Whether to overwrite document if it exists.
        update_schema : bool
            Whether the api should check the documents for vector datatype to update the schema.

        """

        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/documents/insert",
            method="POST",
            parameters={
                "document": document,
                "insert_date": insert_date,
                "overwrite": overwrite,
                "update_schema": update_schema,
            },
        )

    log = insert

    def bulk_insert(
        self,
        dataset_id: str,
        documents: List,
        insert_date: bool = True,
        overwrite: bool = True,
        update_schema: bool = True,
        field_transformers: Optional[List] = None,
        return_documents: bool = False,
        ingest_in_background: bool = True,
    ):
        """
        Documentation can be found here: https://ingest-api-dev-aueast.relevance.ai/latest/documentation#operation/InsertEncode

        - When inserting the document you can optionally specify your own id for a document by using the field name "_id", if not specified a random id is assigned.
        - When inserting or specifying vectors in a document use the suffix (ends with) "_vector_" for the field name. e.g. "product_description_vector_".
        - When inserting or specifying chunks in a document the suffix (ends with) "_chunk_" for the field name. e.g. "products_chunk_".
        - When inserting or specifying chunk vectors in a document's chunks use the suffix (ends with) "_chunkvector_" for the field name. e.g. "products_chunk_.product_description_chunkvector_".
        - Try to keep each batch of documents to insert under 200mb to avoid the insert timing out.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        documents : list
            A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
        insert_date : bool
            Whether to include insert date as a field 'insert_date_'.
        overwrite : bool
            Whether to overwrite document if it exists.
        update_schema : bool
            Whether the api should check the documents for vector datatype to update the schema.
        include_inserted_ids: bool
            Include the inserted IDs in the response
        field_transformers: list
            An example field_transformers object:

            >>> {
            >>>    "field": "string",
            >>>    "output_field": "string",
            >>>    "remove_html": true,
            >>>    "split_sentences": true
            >>> }
        """
        field_transformers = [] if field_transformers is None else field_transformers

        base_url = self.base_url

        if return_documents is False:
            return self.make_http_request(
                endpoint=f"/datasets/{dataset_id}/documents/bulk_insert",
                base_url=base_url,
                method="POST",
                parameters={
                    "documents": documents,
                    "insert_date": insert_date,
                    "overwrite": overwrite,
                    "update_schema": update_schema,
                    "field_transformers": field_transformers,
                    "ingest_in_background": ingest_in_background,
                },
            )

        else:
            response_json = self.make_http_request(
                endpoint=f"/datasets/{dataset_id}/documents/bulk_insert",
                base_url=base_url,
                method="POST",
                parameters={
                    "documents": documents,
                    "insert_date": insert_date,
                    "overwrite": overwrite,
                    "update_schema": update_schema,
                    "field_transformers": field_transformers,
                    "ingest_in_background": ingest_in_background,
                },
            )

            try:
                status_code = response_json.status_code
            except:
                status_code = 200

            return {
                "response_json": response_json,
                "documents": documents,
                "status_code": status_code,
            }

    async def bulk_insert_async(
        self,
        dataset_id: str,
        documents: list,
        insert_date: bool = True,
        overwrite: bool = True,
        update_schema: bool = True,
        field_transformers: Optional[list] = None,
    ):
        """
        Asynchronous version of bulk_insert. See bulk_insert for details.

        Parameters
        ----------
        dataset_id: str
            Unique name of dataset

        documents: list
            A list of documents. A document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'

        insert_date: bool
            Whether to include insert date as a field 'insert_date_'.

        overwrite: bool
            Whether to overwrite document if it exists.

        update_schema: bool
            Whether the api should check the documents for vector datatype to update the schema.

        field_transformers: list
        """
        field_transformers = [] if field_transformers is None else field_transformers

        return await self.make_async_http_request(
            base_url=self.base_url,
            endpoint=f"/datasets/{dataset_id}/documents/bulk_insert",
            method="POST",
            parameters={
                "documents": documents,
                "insert_date": insert_date,
                "overwrite": overwrite,
                "update_schema": update_schema,
                "field_transformers": field_transformers,
            },
        )

    def delete(self, dataset_id: str, confirm: bool = False):
        """
        Delete a dataset

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset

        """
        if confirm:
            # confirm with the user
            self.logger.critical(f"You are about to delete {dataset_id}")
            user_input = input("Confirm? [Y/N] ")
        else:
            user_input = "y"
        # input validation
        if user_input.lower() in ("y", "yes"):
            if "gateway-api-aueast" in self.base_url:
                return self.make_http_request(
                    endpoint=f"/datasets/delete",
                    method="POST",
                    parameters={"dataset_id": dataset_id},
                    raise_error=False,
                )
            else:
                return self.make_http_request(
                    endpoint=f"/datasets/{dataset_id}/delete",
                    method="POST",
                    raise_error=False
                    # parameters={"dataset_id": dataset_id},
                )

        elif user_input.lower() in ("n", "no"):
            self.logger.critical(f"{dataset_id} not deleted")
            return

        else:
            self.logger.critical(f"Error: Input {user_input} unrecognised.")
            return

    def clone(
        self,
        old_dataset: str,
        new_dataset: str,
        schema: Optional[dict] = None,
        rename_fields: Optional[dict] = None,
        remove_fields: Optional[list] = None,
        filters: Optional[list] = None,
    ):
        """
        Clone a dataset into a new dataset. You can use this to rename fields and change data schemas. This is considered a project job.

        Parameters
        ----------
        old_dataset : string
            Unique name of old dataset to copy from
        new_dataset : string
            Unique name of new dataset to copy to
        schema : dict
            Schema for specifying the field that are vectors and its length
        rename_fields : dict
            Fields to rename {'old_field': 'new_field'}. Defaults to no renames
        remove_fields : list
            Fields to remove ['random_field', 'another_random_field']. Defaults to no removes
        filters : list
            Query for filtering the search results
        """
        schema = {} if schema is None else schema
        rename_fields = {} if rename_fields is None else rename_fields
        remove_fields = [] if remove_fields is None else remove_fields
        filters = [] if filters is None else filters

        dataset_id = old_dataset
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/clone",
            method="POST",
            parameters={
                "new_dataset_id": new_dataset,
                "schema": schema,
                "rename_fields": rename_fields,
                "remove_fields": remove_fields,
                "filters": filters,
            },
        )

    def search(
        self,
        query,
        sort_by_created_at_date: bool = False,
        asc: bool = False,
    ):
        """
        Search datasets by their names with a traditional keyword search.

        Parameters
        ----------
        query : string
            Any string that belongs to part of a dataset.
        sort_by_created_at_date : bool
            Sort by created at date. By default shows the newest datasets. Set asc=False to get oldest dataset.
        asc : bool
            Whether to sort results by ascending or descending order
        """
        return self.make_http_request(
            endpoint="/datasets/search",
            method="GET",
            parameters={
                "query": query,
                "sort_by_created_at_date": sort_by_created_at_date,
                "asc": asc,
            },
        )

    def task_status(self, dataset_id: str, task_id: str):
        """
        Check the status of an existing encoding task on the given dataset. \n

        The required task_id was returned in the original encoding request such as datasets.vectorize.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        task_id : string
            The task ID of the earlier queued vectorize task

        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/task_status",
            method="GET",
            parameters={"task_id": task_id},
        )

    def get_file_upload_urls(self, dataset_id: str, files: List):
        """
        Specify a list of file paths. For each file path, a url upload_url is returned. files can be POSTed on upload_url to upload them. They can then be accessed on url. Upon dataset deletion, these files will be deleted.

        Parameters
        -------------
        files: list
            List of files to be uploaded
        dataset_id: str
            The dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/get_file_upload_urls",
            method="POST",
            parameters={"files": files},
        )

    def details(
        self,
        dataset_id: str,
        include_schema: bool = True,
        include_stats: bool = True,
        include_metadata: bool = True,
        include_schema_stats: bool = False,
        include_vector_health: bool = False,
        include_active_jobs: bool = False,
        include_settings: bool = False,
    ):
        """
        Get details about your dataset.

        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/details",
            method="POST",
            parameters={
                "include_schema": include_schema,
                "include_stats": include_stats,
                "include_metadata": include_metadata,
                "include_schema_stats": include_schema_stats,
                "include_vector_health": include_vector_health,
                "include_active_jobs": include_active_jobs,
                "include_settings": include_settings,
            },
        )

    def aggregate(
        self,
        dataset_id: str,
        groupby: List = None,
        metrics: List = None,
        select_fields: List = None,
        sort: List[str] = None,
        asc: bool = False,
        filters: List = None,
        page_size: int = 20,
        page: int = 1,
        aggregation_query: dict = None,
    ):
        """
        Aggregation/Groupby of a collection using an aggregation query. The aggregation query is a json body that follows the schema of:

        .. code-block::

            {
                "groupby" : [
                    {"name": <alias>, "field": <field in the collection>, "agg": "category"},
                    {"name": <alias>, "field": <another groupby field in the collection>, "agg": "numeric"}
                ],
                "metrics" : [
                    {"name": <alias>, "field": <numeric field in the collection>, "agg": "avg"}
                    {"name": <alias>, "field": <another numeric field in the collection>, "agg": "max"}
                ]
            }
            For example, one can use the following aggregations to group score based on region and player name.
            {
                "groupby" : [
                    {"name": "region", "field": "player_region", "agg": "category"},
                    {"name": "player_name", "field": "name", "agg": "category"}
                ],
                "metrics" : [
                    {"name": "average_score", "field": "final_score", "agg": "avg"},
                    {"name": "max_score", "field": "final_score", "agg": "max"},
                    {'name':'total_score','field':"final_score", 'agg':'sum'},
                    {'name':'average_deaths','field':"final_deaths", 'agg':'avg'},
                    {'name':'highest_deaths','field':"final_deaths", 'agg':'max'},
                ]
            }


        """
        # "https://api-dev.ap-southeast-2.relevance.ai/latest/datasets/{DATASET_ID}/aggregate"
        filters = [] if filters is None else filters

        if aggregation_query is None:
            if metrics is None:
                metrics = []
            aggregation_query = {"metrics": metrics}
            if groupby:
                aggregation_query["groupby"] = groupby

            if sort:
                aggregation_query["sort"] = sort

        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/aggregate",
            method="POST",
            parameters={
                "aggregation_query": aggregation_query,
                "filters": filters,
                "page_size": page_size,
                "page": page,
                "asc": asc,
                "select_fields": select_fields,
            },
        )

    def fast_search(
        self,
        dataset_id: str,
        query: str = None,
        queryConfig: dict = None,
        vectorSearchQuery: dict = None,
        instantAnswerQuery: dict = None,
        fieldsToSearch: List = None,
        page: int = 0,
        pageSize: int = 10,
        minimumRelevance: int = 0,
        includeRelevance: bool = True,
        includeDataset: bool = False,
        cleanPayloadUsingSchema: bool = True,
        sort: dict = None,
        includeFields: Optional[List] = None,
        excludeFields: Optional[List] = None,
        includeVectors: bool = True,
        textSort: Optional[dict] = None,
        fieldsToAggregate: Optional[List] = None,
        fieldsToAggregateStats: Optional[List] = None,
        filters: Optional[List] = None,
        relevanceBoosters: Optional[List] = None,
        afterId: Optional[List] = None,
    ):
        """
        Parameters
        ------------

        query: str
            Search for documents that contain this query string in your dataset.
            Use fieldsToSearch parameter to restrict which fields are searched.
        queryConfig: dict
            Configuration for traditional search query.
            Increases or decreases the impact of traditional search when calculating a documents _relevance. new_traditional_relevance = traditional_relevance*queryConfig.weight
        vectorSearchQuery: dict
            Vector search queries.
        instantAnswerQuery: dict
            Provides an instant answer
        fieldsToSearch: list
            The list of fields to search

        """
        # fast search
        parameters = {
            "page": page,
            "pageSize": pageSize,
            "minimumRelevance": minimumRelevance,
            "includeRelevance": includeRelevance,
            "includeDataset": includeDataset,
            "cleanPayloadUsingSchema": cleanPayloadUsingSchema,
            "includeVectors": includeVectors,
        }
        # mypy triggered a lot of really annoying errors that didn't make sense here
        # hmph
        if query is not None:
            parameters["query"] = query  # type: ignore
        if queryConfig is not None:
            parameters["queryConfig"] = queryConfig  # type: ignore
        if vectorSearchQuery is not None:
            parameters["vectorSearchQuery"] = vectorSearchQuery  # type: ignore
        if instantAnswerQuery is not None:
            parameters["instantAnswerQuery"] = instantAnswerQuery  # type: ignore
        if fieldsToSearch is not None:
            parameters["fieldsToSearch"] = fieldsToSearch  # type: ignore
        if sort is not None:
            parameters["sort"] = sort  # type: ignore
        if includeFields is not None:
            parameters["includeFields"] = includeFields  # type: ignore
        if excludeFields is not None:
            parameters["excludeFields"] = excludeFields  # type: ignore
        if textSort is not None:
            parameters["textSort"] = textSort  # type: ignore
        if fieldsToAggregate is not None:
            parameters["fieldsToAggregate"] = fieldsToAggregate  # type: ignore
        if fieldsToAggregateStats is not None:
            parameters["fieldsToAggregateStats"] = fieldsToAggregateStats  # type: ignore
        if filters is not None:
            parameters["filters"] = filters  # type: ignore
        if relevanceBoosters is not None:
            parameters["relevanceBoosters"] = relevanceBoosters  # type: ignore
        if afterId is not None:
            parameters["afterId"] = afterId  # type: ignore
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/simple_search",
            method="POST",
            parameters=parameters,
        )

    def recommend(self, dataset_id, documents_to_recommend: list):
        """
        Recommend documents similar to specific documents. Specify which vector field must be used for recommendation using the documentsToRecommend property.
        Parameters
        ------------
        documentsToRecommend: list
            This takes a list of objects. Each object must specify the id of the document to generate recommendations for, and the vector field that will be compared. Weight can be changed to increase or decrease how much a document contributes to the recommendation. A negative weight will make a document less likely to be recommended.
            `field` - The vector field used for recommendation.
            `id` - The id of the document used for recommendation.
            `weight` - Influences how much a document affects recommendation results. A negative weight causes documents like this to show up less.
        """
        parameters = {"documentsToRecommend": documents_to_recommend}
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/recommend",
            method="POST",
            parameters=parameters,
        )

    def get_settings(self, dataset_id):
        """
        Get settings for a dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/settings",
            method="GET",
            parameters={},
        )

    def post_settings(self, dataset_id, settings: dict):
        """
        Update settings
        """
        parameters = {"settings": settings}
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/settings",
            method="POST",
            parameters=parameters,
        )
