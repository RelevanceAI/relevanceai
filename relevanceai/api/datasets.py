"""All Dataset related functions
"""
from typing import Union, Optional

from relevanceai.base import Base
from relevanceai.api.documents import Documents
from relevanceai.api.monitor import Monitor
from relevanceai.api.tasks import Tasks


class Datasets(Base):
    """All dataset-related functions"""

    def __init__(self, project: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.project = project
        self.api_key = api_key
        self.tasks = Tasks(project=project, api_key=api_key, base_url=base_url)
        self.documents = Documents(project=project, api_key=api_key, base_url=base_url)
        self.monitor = Monitor(project=project, api_key=api_key, base_url=base_url)

        super().__init__(project, api_key, base_url)

    def schema(
        self, dataset_id: str, output_format: str = "json", verbose: bool = True
    ):
        """ 
        Returns the schema of a dataset. Refer to datasets.create for different field types available in a VecDB schema.
        
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/schema",
            method="GET",
            output_format=output_format,
            verbose=verbose,
        )

    def metadata(
        self, dataset_id: str, output_format: str = "json", verbose: bool = True
    ):
        """ 
        Retreives metadata about a dataset. Notably description, data source, etc

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/metadata",
            method="GET",
            output_format=output_format,
            verbose=verbose,
        )

    def create(
        self,
        dataset_id: str,
        schema: dict = {},
        output_format: Union[str, bool] = "json",
        verbose: bool = True,
    ):
        """ 
        A dataset can store documents to be searched, retrieved, filtered and aggregated (similar to Collections in MongoDB, Tables in SQL, Indexes in ElasticSearch). 
        A powerful and core feature of VecDB is that you can store both your metadata and vectors in the same document. When specifying the schema of a dataset and inserting your own vector use the suffix (ends with) "_vector_" for the field name, and specify the length of the vector in dataset_schema. \n

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

        You don't have to specify the schema of every single field when creating a dataset, as VecDB will automatically detect the appropriate data type for each field (vectors will be automatically identified by its "_vector_" suffix). Infact you also don't always have to use this endpoint to create a dataset as /datasets/bulk_insert will infer and create the dataset and schema as you insert new documents. \n

        Note:

            - A dataset name/id can only contain undercase letters, dash, underscore and numbers.
            - "_id" is reserved as the key and id of a document.
            - Once a schema is set for a dataset it cannot be altered. If it has to be altered, utlise the copy dataset endpoint.

        For more information about vectors check out the 'Vectorizing' section, services.search.vector or out blog at https://relevance.ai/blog. For more information about chunks and chunk vectors check out services.search.chunk.

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        schema : dict
            Schema for specifying the field that are vectors and its length
    
        """
        return self.make_http_request(
            endpoint=f"/datasets/create",
            method="POST",
            parameters={"id": dataset_id, "schema": schema},
            output_format=output_format,
            verbose=verbose,
        )

    def list(self, output_format: Optional[str] = "json", verbose: bool = True, retries=None):
        """ List all datasets in a project that you are authorized to read/write. """
        return self.make_http_request(
            endpoint="/datasets/list",
            method="GET",
            output_format=output_format,
            verbose=verbose,
            retries=retries,
        )

    def list_all(
        self,
        include_schema: bool = True,
        include_stats: bool = True,
        include_metadata: bool = True,
        include_schema_stats: bool = False,
        include_vector_health: bool = False,
        include_active_jobs: bool = False,
        dataset_ids: list = [],
        sort_by_created_at_date: bool = False,
        asc: bool = False,
        page_size: int = 20,
        page: int = 1,
        output_format: str = "json",
        verbose: bool = True,
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
            output_format=output_format,
            verbose=verbose,
        )

    def facets(
        self,
        dataset_id,
        fields: list = [],
        date_interval: str = "monthly",
        page_size: int = 5,
        page: int = 1,
        asc: bool = False,
        output_format: str = "json",
        verbose: bool = True,
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
            output_format=output_format,
            verbose=verbose,
        )

    def check_missing_ids(
        self, dataset_id, ids, output_format: str = "json", verbose: bool = True
    ):

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
                output_format=output_format,
                verbose=verbose,
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
        verbose: bool = True,
        retries: int = None,
        output_format: str = "json"
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
        documents : list
            A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
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
            output_format=output_format,
            retries=retries,
            verbose=verbose,
        )


    def bulk_insert(
        self,
        dataset_id: str,
        documents: list,
        insert_date: bool = True,
        overwrite: bool = True,
        update_schema: bool = True,
        field_transformers=[],
        verbose: bool = True,
        return_documents: bool = False,
        retries: int = None,
        output_format: str = "json",
        base_url="https://ingest-api-dev-aueast.relevance.ai/latest",
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
                },
                output_format=output_format,
                retries=retries,
                verbose=verbose,
            )

        else:
            insert_response = self.make_http_request(
                endpoint=f"/datasets/{dataset_id}/documents/bulk_insert",
                base_url=base_url,
                method="POST",
                parameters={
                    "documents": documents,
                    "insert_date": insert_date,
                    "overwrite": overwrite,
                    "update_schema": update_schema,
                    "field_transformers": field_transformers,
                },
                output_format=False,
                retries=retries,
                verbose=verbose,
            )

            try:
                response_json = insert_response.json()
            except:
                response_json = None

            return {
                "response_json": response_json,
                "documents": documents,
                "status_code": insert_response.status_code,
            }

    def delete(
        self,
        dataset_id: str,
        confirm: bool = False,
        output_format: str = "json",
        verbose: bool = True,
    ):
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
            return self.make_http_request(
                endpoint=f"/datasets/delete",
                method="POST",
                parameters={"dataset_id": dataset_id},
                output_format=output_format,
                verbose=verbose,
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
        schema: dict = {},
        rename_fields: dict = {},
        remove_fields: list = [],
        filters: list = [],
        output_format: str = "json",
        verbose: bool = True,
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
            output_format=output_format,
            verbose=verbose,
        )

    def search(
        self,
        query,
        sort_by_created_at_date: bool = False,
        asc: bool = False,
        output_format: str = "json",
        verbose: bool = True,
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
            output_format=output_format,
            verbose=verbose,
        )

    def vectorize(
        self,
        dataset_id: str,
        model_id: str,
        fields: list = [],
        filters: list = [],
        refresh: bool = False,
        alias: str = "default",
        chunksize: int = 20,
        chunk_field: str = None,
        output_format: str = "json",
        verbose: bool = True,
    ):
        """
        Queue the encoding of a dataset using the method given by model_id.
        
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        model_id : string
            Model ID to use for vectorizing (encoding.)
        fields : list
            Fields to remove ['random_field', 'another_random_field']. Defaults to no removes
        filters : list
            Filters to run against
        refresh : bool
            If True, re-runs encoding on whole dataset.
        alias : string
            Alias used to name a vector field. Belongs in field_{alias}vector
        chunksize : int
            Batch for each encoding. Change at your own risk.
        chunk_field : string
            The chunk field. If the chunk field is specified, the field to be encoded should not include the chunk field.
        
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/vectorize",
            method="GET",
            parameters={
                "model_id": model_id,
                "fields": fields,
                "filters": filters,
                "refresh": refresh,
                "alias": alias,
                "chunksize": chunksize,
                "chunk_field": chunk_field,
            },
            output_format=output_format,
            verbose=verbose,
        )

    def task_status(
        self,
        dataset_id: str,
        task_id: str,
        output_format: str = "json",
        verbose: bool = True,
    ):
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
            parameters={
                "task_id": task_id
            },
            output_format=output_format,
            verbose=verbose,
        )


