# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
import warnings
import requests
import pandas as pd

from doc_utils import DocUtils
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from tqdm.auto import tqdm

from relevanceai._api import APIClient

from relevanceai.utils.logger import FileLogger
from relevanceai.utils.decorators.analytics import track
from relevanceai.utils import make_id

from relevanceai.constants.warning import Warning


class Write(APIClient):
    @track
    def insert_documents(
        self,
        documents: list,
        bulk_fn: Callable = None,
        max_workers: int = 8,
        retry_chunk_mult: float = 0.5,
        show_progress_bar: bool = False,
        chunksize: int = 0,
        use_json_encoder: bool = True,
        create_id: bool = False,
        **kwargs,
    ) -> Dict:

        """
        Insert a list of documents with multi-threading automatically enabled.

        - When inserting the document you can optionally specify your own id for a document by using the field name "_id", if not specified a random id is assigned.
        - When inserting or specifying vectors in a document use the suffix (ends with) "_vector_" for the field name. e.g. "product_description_vector_".
        - When inserting or specifying chunks in a document the suffix (ends with) "_chunk_" for the field name. e.g. "products_chunk_".
        - When inserting or specifying chunk vectors in a document's chunks use the suffix (ends with) "_chunkvector_" for the field name. e.g. "products_chunk_.product_description_chunkvector_".

        Documentation can be found here: https://ingest-api-dev-aueast.relevance.ai/latest/documentation#operation/InsertEncode

        Parameters
        ----------
        documents: list
            A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
        bulk_fn : callable
            Function to apply to documents before uploading
        max_workers : int
            Number of workers active for multi-threading
        retry_chunk_mult: int
            Multiplier to apply to chunksize if upload fails
        chunksize : int
            Number of documents to upload per worker. If None, it will default to the size specified in config.upload.target_chunk_mb
        use_json_encoder : bool
            Whether to automatically convert documents to json encodable format

        Example
        --------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            documents = [
                {
                    "_id": "10",
                    "value": 5
                },
                {
                    "_id": "332",
                    "value": 10
                }
            ]

            df.insert_documents(documents)

        """
        results = self._insert_documents(
            dataset_id=self.dataset_id,
            documents=documents,
            bulk_fn=bulk_fn,
            max_workers=max_workers,
            retry_chunk_mult=retry_chunk_mult,
            show_progress_bar=show_progress_bar,
            chunksize=chunksize,
            use_json_encoder=use_json_encoder,
            create_id=create_id,
            **kwargs,
        )
        return self._process_insert_results(results)

    @track
    def insert_csv(
        self,
        filepath_or_buffer,
        chunksize: int = 10000,
        max_workers: int = 8,
        retry_chunk_mult: float = 0.5,
        show_progress_bar: bool = False,
        index_col: int = None,
        csv_args: Optional[dict] = None,
        col_for_id: str = None,
        auto_generate_id: bool = True,
    ) -> Dict:
        """
        Insert data from csv file

        Parameters
        ----------
        filepath_or_buffer :
            Any valid string path is acceptable. The string could be a URL. Valid URL schemes include http, ftp, s3, gs, and file.
        chunksize : int
            Number of lines to read from csv per iteration
        max_workers : int
            Number of workers active for multi-threading
        retry_chunk_mult: int
            Multiplier to apply to chunksize if upload fails
        csv_args : dict
            Optional arguments to use when reading in csv. For more info, see https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
        index_col : None
            Optional argument to specify if there is an index column to be skipped (e.g. index_col = 0)
        col_for_id : str
            Optional argument to use when a specific field is supposed to be used as the unique identifier ('_id')
        auto_generate_id: bool = True
            Automatically generateds UUID if auto_generate_id is True and if the '_id' field does not exist

        Example
        ---------
        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")

            csv_filename = "temp.csv"
            df.insert_csv(csv_filename)

        """
        csv_args = {} if csv_args is None else csv_args

        results = self._insert_csv(
            dataset_id=self.dataset_id,
            filepath_or_buffer=filepath_or_buffer,
            chunksize=chunksize,
            max_workers=max_workers,
            retry_chunk_mult=retry_chunk_mult,
            show_progress_bar=show_progress_bar,
            index_col=index_col,
            csv_args=csv_args,
            col_for_id=col_for_id,
            auto_generate_id=auto_generate_id,
        )
        self._process_insert_results(results)
        return results

    @track
    def insert_pandas_dataframe(
        self, df: pd.DataFrame, col_for_id=None, *args, **kwargs
    ):
        """
        Insert a dataframe into the dataset.
        Takes additional args and kwargs based on `insert_documents`.

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample_dataset_id")
            pandas_df = pd.DataFrame({"value": [3, 2, 1], "_id": ["10", "11", "12"]})
            df.insert_pandas_dataframe(pandas_df)

        """
        if col_for_id is not None:
            df["_id"] = df[col_for_id]

        else:
            uuids = [make_id(df.iloc[index]) for index in range(len(df))]
            df["_id"] = uuids

        def _is_valid(v):
            try:
                if pd.isna(v):
                    return False
                else:
                    return True
            except:
                return True

        documents = [
            {k: v for k, v in doc.items() if _is_valid(v)}
            for doc in df.to_dict(orient="records")
        ]

        results = self._insert_documents(self.dataset_id, documents, *args, **kwargs)
        self.print_search_dashboard_url(self.dataset_id)
        return results

    def insert_media_folder(
        self,
        path: Union[Path, str],
        field: str = "medias",
        recurse: bool = True,
        *args,
        **kwargs,
    ):
        """
        Given a path to a directory, this method loads all media-related files
        into a Dataset.

        Parameters
        ----------
        field: str
            A text field of a dataset.

        path: Union[Path, str]
            The path to the directory containing medias.

        recurse: bool
            Indicator that determines whether to recursively insert medias from
            subdirectories in the directory.

        Returns
        -------
            dict

        Example
        -------
        .. code-block::

            from relevanceai import Client
            client = Client()
            ds = client.Dataset("dataset_id")

            from pathlib import Path
            path = Path("medias/")
            # list(path.iterdir()) returns
            # [
            #    PosixPath('media.jpg'),
            #    PosixPath('more-medias'), # a directory
            # ]

            get_all_medias: bool = True
            if get_all_medias:
                # Inserts all medias, even those in the more-medias directory
                ds.insert_media_folder(
                    field="medias", path=path, recurse=True
                )
            else:
                # Only inserts media.jpg
                ds.insert_media_folder(
                    field="medias", path=path, recurse=False
                )

        """
        if isinstance(path, str):
            path = Path(path)

            if not path.is_dir():
                raise Exception(f"{path} is not a proper path")

        from mimetypes import types_map

        media_extensions = set(
            k.lower() for k, v in types_map.items() if v.startswith("media/")
        )

        def get_paths(path: Path, medias: List[str]) -> List[str]:
            for file in path.iterdir():
                if file.is_dir() and recurse:
                    medias.extend(get_paths(file, []))
                elif file.is_file() and file.suffix.lower() in media_extensions:
                    medias.append(str(file))
                else:
                    continue

            return medias

        medias = get_paths(path, [])
        documents = list(
            map(
                lambda media: {"_id": make_id(media), "path": media, field: media},
                medias,
            )
        )
        results = self.insert_documents(documents, *args, **kwargs)
        self.image_fields.append(field)
        return results

    @track
    def upsert_documents(
        self,
        documents: list,
        bulk_fn: Callable = None,
        max_workers: int = 8,
        retry_chunk_mult: float = 0.5,
        chunksize: int = 0,
        show_progress_bar=False,
        use_json_encoder: bool = True,
        return_json: bool = False,
        create_id: bool = False,
    ) -> Dict:

        """
        Update a list of documents with multi-threading automatically enabled.
        Edits documents by providing a key value pair of fields you are adding or changing, make sure to include the "_id" in the documents.


        Parameters
        ----------
        documents : list
            A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
        bulk_fn : callable
            Function to apply to documents before uploading
        max_workers : int
            Number of workers active for multi-threading
        retry_chunk_mult: int
            Multiplier to apply to chunksize if upload fails
        chunksize : int
            Number of documents to upload per worker. If None, it will default to the size specified in config.upload.target_chunk_mb
        use_json_encoder : bool
            Whether to automatically convert documents to json encodable format
        create_id: bool
            If True, creates ID for users automatically

        Example
        ----------
        .. code-block::

            from relevanceai import Client

            client = Client()

            documents = [
                {
                    "_id": "321",
                    "value": 10
                },
                {
                    "_id": "4243",
                    "value": 100
                }
            ]

            dataset_id = "sample_dataset_id"
            ds = client.Dataset(dataset_id)
            ds.upsert_documents(documents)

        """
        results = self._update_documents(
            self.dataset_id,
            documents=documents,
            bulk_fn=bulk_fn,
            max_workers=max_workers,
            retry_chunk_mult=retry_chunk_mult,
            show_progress_bar=show_progress_bar,
            chunksize=chunksize,
            use_json_encoder=use_json_encoder,
            create_id=create_id,
        )
        return self._process_insert_results(results, return_json=return_json)

    @track
    def apply(
        self,
        func: Callable,
        retrieve_chunksize: int = 100,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        show_progress_bar: bool = True,
        use_json_encoder: bool = True,
        axis: int = 0,
        log_to_file: bool = True,
        log_file: Optional[str] = None,
        **apply_args,
    ):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is either the DataFrame’s index (axis=0) or the DataFrame’s columns (axis=1). By default (result_type=None), the final return type is inferred from the return type of the applied function. Otherwise, it depends on the result_type argument.

        Parameters
        --------------
        func: function
            Function to apply to each document
        retrieve_chunksize: int
            The number of documents that are received from the original collection with each loop iteration.
        max_workers: int
            The number of processors you want to parallelize with
        max_error: int
            How many failed uploads before the function breaks
        json_encoder : bool
            Whether to automatically convert documents to json encodable format
        axis: int
            Axis along which the function is applied.
            - 9 or 'index': apply function to each column
            - 1 or 'columns': apply function to each row

        Example
        ---------
        .. code-block::

            from relevanceai import Client
            from relevanceai.package_utils.datasets import mock_documents

            client = Client()

            ds = client.Dataset("sample_dataset_id")
            ds.upsert_documents(mock_documents(100))

            def update_doc(doc):
                doc["value"] = 2
                return doc

            df.apply(update_doc)

            def update_doc_wargs(doc, value1, value2):
                doc["value"] += value1
                doc["value"] *= value2
                return doc

            df.apply(func=update_doc, value1=3, value2=2)

        """
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        if axis == 1:
            raise ValueError("We do not support column-wise operations!")

        def bulk_fn(documents):
            new_documents = []
            for d in documents:
                new_d = func(d, **apply_args)
                new_documents.append(new_d)
            return documents

        results = self.pull_update_push_async(
            self.dataset_id,
            bulk_fn,
            retrieve_chunk_size=retrieve_chunksize,
            filters=filters,
            select_fields=select_fields,
            show_progress_bar=show_progress_bar,
            use_json_encoder=use_json_encoder,
            log_to_file=log_to_file,
            log_file=log_file,
        )
        if results is None:
            print("✅ Successfully ran!")
            return
        for k, v in results.items():
            if v != []:
                print("️❗❗Errors detected when running apply.")
                return results
        print("✅ Successfully ran!")

    @track
    def bulk_apply(
        self,
        bulk_func: Callable,
        retrieve_chunksize: int = 100,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        show_progress_bar: bool = True,
        use_json_encoder: bool = True,
        log_to_file: bool = True,
    ):
        """
        Apply a bulk function along an axis of the DataFrame.

        Parameters
        ------------
        bulk_func: function
            Function to apply to a bunch of documents at a time
        retrieve_chunksize: int
            The number of documents that are received from the original collection with each loop iteration.
        max_workers: int
            The number of processors you want to parallelize with
        max_error: int
            How many failed uploads before the function breaks
        json_encoder : bool
            Whether to automatically convert documents to json encodable format
        axis: int
            Axis along which the function is applied.
            - 9 or 'index': apply function to each column
            - 1 or 'columns': apply function to each row

        Example
        ---------
        .. code-block::

            from relevanceai import Client

            client = Client()

            df = client.Dataset("sample_dataset_id")

            def update_documents(documents):
                for d in documents:
                    d["value"] = 10
                return documents

            df.apply(update_documents)
        """
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        return self.pull_update_push_async(
            self.dataset_id,
            bulk_func,
            retrieve_chunk_size=retrieve_chunksize,
            filters=filters,
            select_fields=select_fields,
            show_progress_bar=show_progress_bar,
            use_json_encoder=use_json_encoder,
            log_to_file=log_to_file,
        )

    @track
    def cat(self, vector_name: Union[str, None] = None, fields: Optional[List] = None):
        """
        Concatenates numerical fields along an axis and reuploads this vector for other operations

        Parameters
        ----------
        vector_name: str, default None
            name of the new concatenated vector field
        fields: List
            fields alone which the new vector will concatenate

        Example
        -----------------
        .. code-block::

            from relevanceai import Client

            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)

            fields = [
                "numeric_field1",
                "numeric_field2",
                "numeric_field3"
            ]

            df.concat(fields)

            concat_vector_field_name = "concat_vector_"
            df.concat(vector_name=concat_vector_field_name, fields=fields)
        """
        fields = [] if fields is None else fields

        if vector_name is None:
            vector_name = "_".join(fields) + "_cat_vector_"

        def cat_fields(documents, field_name):
            cat_vector_documents = [
                {"_id": sample["_id"], field_name: [sample[field] for field in fields]}
                for sample in documents
            ]
            return cat_vector_documents

        self.pull_update_push(
            self.dataset_id, cat_fields, updating_args={"field_name": vector_name}
        )

    concat = cat

    def _label_cluster(self, label: Union[int, str]):
        if isinstance(label, (int, float)):
            return "cluster-" + str(label)
        return str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]

    def set_cluster_labels(self, vector_fields, alias, labels):
        def add_cluster_labels(documents):
            documents = self.get_all_documents(self.dataset_id)
            documents = list(filter(DocUtils.list_doc_fields, documents))
            set_cluster_field = (
                "_cluster_" + ".".join(vector_fields).lower() + "." + alias
            )
            self.set_field_across_documents(
                set_cluster_field,
                self._label_clusters(list(labels)),
                documents,
            )
            return documents

        self.pull_update_push(self.dataset_id, add_cluster_labels)

    @track
    def create(self, schema: Optional[dict] = None) -> Dict:
        """
        A dataset can store documents to be searched, retrieved, filtered and aggregated (similar to Collections in MongoDB, Tables in SQL, Indexes in ElasticSearch).
        A powerful and core feature of VecDB is that you can store both your metadata and vectors in the same document. When specifying the schema of a dataset and inserting your own vector use the suffix (ends with) "_vector_" for the field name, and specify the length of the vector in dataset_schema. \n

        For example:

        .. code-block::
            {
                "product_media_vector_": 1024,
                "product_text_description_vector_" : 128
            }

        These are the field types supported in our datasets: ["text", "numeric", "date", "dict", "chunks", "vector", "chunkvector"]. \n

        For example:

        .. code-block::

            {
                "product_text_description" : "text",
                "price" : "numeric",
                "created_date" : "date",
                "product_texts_chunk_": "chunks",
                "product_text_chunkvector_" : 1024
            }

        You don't have to specify the schema of every single field when creating a dataset, as VecDB will automatically detect the appropriate data type for each field (vectors will be automatically identified by its "_vector_" suffix). Infact you also don't always have to use this endpoint to create a dataset as /datasets/bulk_insert will infer and create the dataset and schema as you insert new documents. \n

        Note:

            - A dataset name/id can only contain undercase letters, dash, underscore and numbers.
            - "_id" is reserved as the key and id of a document.
            - Once a schema is set for a dataset it cannot be altered. If it has to be altered, utlise the copy dataset endpoint.

        For more information about vectors check out the 'Vectorizing' section, services.search.vector or out blog at https://relevance.ai/blog. For more information about chunks and chunk vectors check out services.search.chunk.

        Parameters
        ----------
        schema : dict
            Schema for specifying the field that are vectors and its length

        Example
        ----------
        .. code-block::

            from relevanceai import Client
            client = Client()

            documents = [
                {
                    "_id": "321",
                    "value": 10
                },
                {
                    "_id": "4243",
                    "value": 100
                }
            ]

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            df.create()

            df.insert_documents(documents)
        """
        schema = {} if schema is None else schema

        return self.datasets.create(self.dataset_id, schema=schema)

    @track
    def delete(self):
        """
        Delete a dataset

        Example
        ---------
        .. code-block::

            from relevanceai import Client
            client = Client()

            dataset_id = "sample_dataset_id"
            df = client.Dataset(dataset_id)
            df.delete()

        """
        return self.datasets.delete(self.dataset_id)

    insert_df = insert_pandas_dataframe

    def _upload_media(
        self, presigned_url: str, media_content: bytes, verbose: bool = True
    ):
        if not isinstance(media_content, bytes):
            raise ValueError(
                f"media needs to be in a bytes format. Currently in {type(media_content)}"
            )
        response = requests.put(presigned_url, data=media_content)
        if response.status_code == 200:
            if verbose:
                print("media successfully uploaded.")

    def insert_media_url(self, media_url: str, verbose: bool = True):
        """
        Insert a single media URL
        """
        # media to download
        response = self.datasets.get_file_upload_urls(
            self.dataset_id, files=[media_url]
        )
        url = response["files"][0]["url"]
        self._upload_media(
            presigned_url=response["files"][0]["upload_url"],
            media_content=requests.get(media_url).content,
        )
        if verbose:
            print(f"media is hosted at {url}")
        return url

    def insert_media_urls(
        self,
        media_urls: List[str],
        verbose: bool = True,
        file_log: str = "insert_media_urls.log",
    ):
        """
        Insert a single media URL
        """
        # media to download
        response = self.datasets.get_file_upload_urls(self.dataset_id, files=media_urls)
        response_docs: dict = {"media_documents": [], "failed_medias": []}
        with FileLogger(file_log):
            for i, im in enumerate(tqdm(media_urls)):
                response_doc = {}
                response_doc["media_file"] = im
                response_doc["media_url"] = response["files"][i]["url"]
                try:
                    self._upload_media(
                        presigned_url=response["files"][i]["upload_url"],
                        media_content=requests.get(im).content,
                        verbose=verbose,
                    )
                    response_docs["media_documents"].append(response_doc)
                except Exception as e:
                    if verbose:
                        print(f"Failed to upload {im}.")
                    if verbose:
                        print(e)
                    response_docs["failed_medias"].append(response_doc)
        # Return the media URLs
        return response_docs

    def _open_local_media(self, fn: str) -> bytes:
        with open(fn, "rb") as fn_byte:
            f = fn_byte.read()
            b = bytes(f)
        return b

    def insert_local_media(self, media_fn: str, verbose: bool = True):
        """
        Insert local media

        Parameters
        -------------
        media_fn: str
            A local media to upload
        verbose: bool
            If True, prints a statement after uploading each media
        """
        # media to download
        response = self.datasets.get_file_upload_urls(self.dataset_id, files=[media_fn])
        url = response["files"][0]["url"]
        self._upload_media(
            presigned_url=response["files"][0]["upload_url"],
            media_content=self._open_local_media(media_fn),
            verbose=verbose,
        )
        if verbose:
            print(f"media is hosted at {url}.")
        return url

    def insert_local_medias(
        self,
        media_fns: List[str],
        verbose: bool = False,
        file_log="local_media_upload.log",
    ):
        """Insert a list of local medias.

        Parameters
        ------------
        media_fns: List[str]
            A list of local medias
        verbose: bool
            If True, this will print after each successful upload.
        file_log: str
            The log to write
        """
        response = self.datasets.get_file_upload_urls(self.dataset_id, files=media_fns)
        response_docs: dict = {"media_documents": [], "failed_medias": []}
        with FileLogger(file_log) as f:
            for i, media_fn in enumerate(tqdm(media_fns)):
                response_doc = {}
                response_doc["media_file"] = media_fn
                response_doc["media_url"] = response["files"][i]["url"]
                try:
                    self._upload_media(
                        presigned_url=response["files"][i]["upload_url"],
                        media_content=self._open_local_media(media_fn),
                        verbose=verbose,
                    )
                    response_docs["media_documents"].append(response_doc)
                except Exception as e:
                    print(f"failed to upload {media_fn}")
                    print(e)
                    response_docs["failed_medias"].append(response_doc)
        return response_docs

    def get_media_documents(
        self,
        media_fns: List[str],
        verbose: bool = False,
        file_log: str = "media_upload.log",
    ) -> dict:
        """
        Bulk insert medias. Returns a link to once it has been hosted

        Parameters
        --------------
        media_fns: List[str]
            List of medias to upload
        verbose: bool
            If True, prints statements after uploading
        file_log: str
            The file log to write
        """
        # Algorithm aims to insert local or hosted medias
        if "http" in media_fns[0]:
            return self.insert_media_urls(media_fns, verbose=verbose, file_log=file_log)
        else:
            return self.insert_local_medias(
                media_fns, verbose=verbose, file_log=file_log
            )

    def upsert_media(
        self,
        media_fns: List[str],
        verbose: bool = False,
        file_log: str = "media_upload.log",
        **kw,
    ):
        """
        Insert medias into a dataset.

        Parameters
        -------------

        media_fns: List[str]
            A list of medias to upsert
        verbose: bool
            If True, prints statements after uploading
        file_log: str
            The file log to write
        """
        documents = self.get_media_documents(
            media_fns=media_fns, verbose=verbose, file_log=file_log
        )
        return self.upsert_documents(documents["media_documents"], create_id=True, **kw)

    def delete_documents(self, document_ids: List[str]):
        """
        Delete documents in a dataset

        Parameters
        ------------
        document_ids: List[str]
            A list of document IDs to delete

        """
        return self.datasets.documents.bulk_delete(
            dataset_id=self.dataset_id, ids=document_ids
        )
