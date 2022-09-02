# -*- coding: utf-8 -*-
"""Batch Insert"""
import os
import math
import time

import warnings
import traceback

import pandas as pd

from ast import literal_eval

from datetime import datetime

from typing import Any, Callable, Dict, List, Optional

from tqdm.auto import tqdm

from relevanceai._api.batch.retrieve import BatchRetrieveClient
from relevanceai._api.batch.local_logger import PullTransformPushLocalLogger

from tqdm.auto import tqdm

from relevanceai.utils import make_id
from relevanceai.utils.concurrency import Push
from relevanceai.utils.helpers.helpers import getsizeof
from relevanceai.utils.logger import FileLogger
from relevanceai.utils.progress_bar import progress_bar
from relevanceai.utils.decorators.version import beta
from relevanceai.utils.decorators.analytics import track

from relevanceai.constants.errors import FieldNotFoundError
from relevanceai.constants.warning import Warning
from relevanceai.constants import ONE_MB


class BatchInsertClient(BatchRetrieveClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _insert_documents(
        self,
        dataset_id: str,
        documents: list,
        max_workers: Optional[int] = 2,
        show_progress_bar: bool = False,
        chunksize: Optional[int] = None,
        overwrite: bool = True,
        ingest_in_background: bool = True,
        verbose: Optional[bool] = False,
    ):

        """
        Insert a list of documents with multi-threading automatically enabled.

        - When inserting the document you can optionally specify your own id for a document by using the field name "_id", if not specified a random id is assigned.
        - When inserting or specifying vectors in a document use the suffix (ends with) "_vector_" for the field name. e.g. "product_description_vector_".
        - When inserting or specifying chunks in a document the suffix (ends with) "_chunk_" for the field name. e.g. "products_chunk_".
        - When inserting or specifying chunk vectors in a document's chunks use the suffix (ends with) "_chunkvector_" for the field name. e.g. "products_chunk_.product_description_chunkvector_".

        Documentation can be found here: https://ingest-api-dev-aueast.relevance.ai/latest/documentation#operation/InsertEncode

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
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

        Example
        --------

        >>> from relevanceai import Client
        >>> client = Client()
        >>> df = client.Dataset("sample_dataset_id")
        >>> documents = [{"_id": "10", "value": 5}, {"_id": "332", "value": 10}]
        >>> df.insert_documents(documents)

        """

        if verbose:
            self.logger.info(f"You are currently inserting into {dataset_id}")
            tracking_message = f"while inserting, you can visit monitor the dataset at https://cloud.relevance.ai/dataset/{dataset_id}/dashboard/monitor/"
            self.logger.info(tracking_message)
            tqdm.write(tracking_message)

        # Check if the collection exists
        self.datasets.create(dataset_id)

        return self._write_documents(
            dataset_id=dataset_id,
            bulk_func=self.datasets.bulk_insert,
            documents=documents,
            max_workers=max_workers,
            show_progress_bar=show_progress_bar,
            chunksize=chunksize,
            overwrite=overwrite,
            ingest_in_background=ingest_in_background,
        )

    def _update_documents(
        self,
        dataset_id: str,
        documents: List[Dict[str, Any]],
        max_workers: Optional[int] = 2,
        show_progress_bar: bool = False,
        chunksize: Optional[int] = None,
        ingest_in_background: bool = True,
        verbose: Optional[bool] = False,
    ):
        """
        Update a list of documents with multi-threading automatically enabled.
        Edits documents by providing a key value pair of fields you are adding or changing, make sure to include the "_id" in the documents.

        Example
        ----------

        >>> from relevanceai import Client
        >>> url = "https://api-aueast.relevance.ai/v1/"
        >>> collection = ""
        >>> project = ""
        >>> api_key = ""
        >>> client = Client(project=project, api_key=api_key, firebase_uid=firebase_uid)
        >>> documents = client.datasets.documents.get_where(collection, select_fields=['title'],
            after_id=True)
        >>> while len(documents['documents']) > 0:
        >>>     documents['documents'] = model.bulk_encode_documents(['product_name'], documents['documents'])
        >>>     client.update_documents(collection, documents['documents'])
        >>>     documents = client.datasets.documents.get_where(collection, select_fields=['product_name'], search_after=documents['after_id'])

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
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
        """

        if verbose:
            self.logger.info(f"You are currently updating {dataset_id}")
            tracking_message = f"while updating, you can visit monitor the dataset at https://cloud.relevance.ai/dataset/{dataset_id}/dashboard/monitor/"
            self.logger.info(tracking_message)
            tqdm.write(tracking_message)

        return self._write_documents(
            dataset_id=dataset_id,
            documents=documents,
            bulk_func=self.datasets.documents.bulk_update,
            max_workers=max_workers,
            show_progress_bar=show_progress_bar,
            chunksize=chunksize,
            ingest_in_background=ingest_in_background,
        )

    update_documents = _update_documents

    def pull_update_push(
        self,
        dataset_id: str,
        update_function,
        updated_dataset_id: str = None,
        log_file: str = None,
        updated_documents_file: str = None,
        updating_args: Optional[dict] = None,
        retrieve_chunk_size: int = 100,
        max_workers: int = 2,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        show_progress_bar: bool = True,
        log_to_file: bool = True,
    ):
        """
        Loops through every document in your collection and applies a function (that is specified by you) to the documents.
        These documents are then uploaded into either an updated collection, or back into the original collection.

        Parameters
        ----------
        dataset_id: string
            The dataset_id of the collection where your original documents are

        update_function: function
            A function created by you that converts documents in your original collection into the updated documents. The function must contain a field which takes in a list of documents from the original collection. The output of the function must be a list of updated documents.

        updated_dataset_id: string
            The dataset_id of the collection where your updated documents are uploaded into. If 'None', then your original collection will be updated.

        log_file: str
            The log file to direct any information or issues that may crop up.
            If no log file is specified, one will automatically be created.

        updated_documents_file: str
            A file to keep track of documents that have already been update.
            If a file is not specified, one will automatically be created.

        updating_args: dict
            Additional arguments to your update_function, if they exist. They must be in the format of {'Argument': Value}

        retrieve_chunk_size: int
            The number of documents that are received from the original collection with each loop iteration.

        max_workers: int
            The number of processors you want to parallelize with

        filters: list
            A list of filters to apply on the retrieval query

        select_fields: list
            A list of fields to query over

        use_json_encoder : bool
            Whether to automatically convert documents to json encodable format
        """
        updating_args = {} if updating_args is None else updating_args
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        if not callable(update_function):
            raise TypeError(
                "Your update function needs to be a function! Please read the documentation if it is not."
            )

        # Check if a logging_collection has been supplied
        if log_file is None:
            log_file = (
                dataset_id
                + "_"
                + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
                + "_pull_update_push"
                + ".log"
            )
            self.logger.info(f"Created {log_file}")

        if updated_documents_file is None:
            updated_documents_file = "_".join(
                [
                    dataset_id,
                    str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S")),
                    "pull_update_push-updated_documents.temp",
                ]
            )
            self.logger.info(f"Created {updated_documents_file}")

        with FileLogger(fn=log_file, verbose=True, log_to_file=log_to_file):
            # Instantiate the logger to document the successful IDs
            PULL_UPDATE_PUSH_LOGGER = PullTransformPushLocalLogger(
                updated_documents_file
            )

            # Track failed documents
            failed_documents: List[Dict] = []
            failed_documents_detailed: List[Dict] = []

            # Track successful documents
            success_documents: List[str] = []

            # Get document lengths to calculate iterations
            original_length = self.get_number_of_documents(dataset_id, filters)

            # get the remaining number in case things break
            remaining_length = (
                original_length - PULL_UPDATE_PUSH_LOGGER.count_ids_in_fn()
            )

            # iterations_required = math.ceil(remaining_length / retrieve_chunk_size)
            iterations_required = math.ceil(remaining_length / retrieve_chunk_size)

            # Get incomplete documents from raw collection
            for _ in progress_bar(
                range(iterations_required), show_progress_bar=show_progress_bar
            ):
                retrieve_filters = filters + [
                    {
                        "field": "ids",
                        "filter_type": "ids",
                        "condition": "!=",
                        "condition_value": success_documents,
                    }
                ]

                orig_json = self.datasets.documents.get_where(
                    dataset_id,
                    filters=retrieve_filters,
                    page_size=retrieve_chunk_size,
                    select_fields=select_fields,
                )

                documents = orig_json["documents"]

                try:
                    updated_data = update_function(documents, **updating_args)
                except Exception as e:
                    self.logger.error("Your updating function does not work: " + str(e))
                    traceback.print_exc()
                    return

                updated_documents = [i["_id"] for i in documents]

                # Upload documents
                if updated_dataset_id is None:
                    insert_json = self._update_documents(
                        dataset_id=dataset_id,
                        documents=updated_data,
                        max_workers=max_workers,
                        show_progress_bar=False,
                    )
                else:
                    insert_json = self._insert_documents(
                        dataset_id=updated_dataset_id,
                        documents=updated_data,
                        max_workers=max_workers,
                        show_progress_bar=False,
                    )

                chunk_failed = insert_json["failed_documents"]
                chunk_documents_detailed = insert_json["failed_documents_detailed"]
                failed_documents.extend(chunk_failed)
                failed_documents_detailed.extend(chunk_documents_detailed)
                success_documents += list(
                    set(updated_documents) - set(failed_documents)
                )
                PULL_UPDATE_PUSH_LOGGER.log_ids(success_documents)
                self.logger.success(
                    f"Chunk of {retrieve_chunk_size} original documents updated and uploaded with {len(chunk_failed)} failed documents!"
                )

            if failed_documents:
                # This will be picked up by FileLogger
                print("The following documents failed to be updated/inserted:")
                for failed_document in failed_documents:
                    print(f"  * {failed_document}")

        self.logger.info(f"Deleting {updated_documents_file}")
        if os.path.exists(updated_documents_file):
            os.remove(updated_documents_file)

        self.logger.success(f"Pull, Update, Push is complete!")

        return {
            "failed_documents": failed_documents,
            "failed_documents_detailed": failed_documents_detailed,
        }

    def pull_update_push_to_cloud(
        self,
        dataset_id: str,
        update_function,
        updated_dataset_id: str = None,
        logging_dataset_id: str = None,
        updating_args: Optional[dict] = None,
        retrieve_chunk_size: int = 100,
        retrieve_chunk_size_failure_retry_multiplier: float = 0.5,
        number_of_retrieve_retries: int = 3,
        max_workers: int = 2,
        max_error: int = 1000,
        filters: Optional[list] = None,
        select_fields: Optional[list] = None,
        show_progress_bar: bool = True,
    ):
        """
        Loops through every document in your collection and applies a function (that is specified by you) to the documents.
        These documents are then uploaded into either an updated collection, or back into the original collection.

        Parameters
        ------------
        original_dataset_id: string
            The dataset_id of the collection where your original documents are
        logging_dataset_id: string
            The dataset_id of the collection which logs which documents have been updated. If 'None', then one will be created for you.
        updated_dataset_id: string
            The dataset_id of the collection where your updated documents are uploaded into. If 'None', then your original collection will be updated.
        update_function: function
            A function created by you that converts documents in your original collection into the updated documents. The function must contain a field which takes in a list of documents from the original collection. The output of the function must be a list of updated documents.
        updating_args: dict
            Additional arguments to your update_function, if they exist. They must be in the format of {'Argument': Value}
        retrieve_chunk_size: int
            The number of documents that are received from the original collection with each loop iteration.
        retrieve_chunk_size_failure_retry_multiplier: int
            If fails, retry on each chunk
        max_workers: int
            The number of processors you want to parallelize with
        max_error: int
            How many failed uploads before the function breaks
        json_encoder : bool
            Whether to automatically convert documents to json encodable format
        """
        updating_args = {} if updating_args is None else updating_args
        filters = [] if filters is None else filters
        select_fields = [] if select_fields is None else select_fields

        # Check if a logging_collection has been supplied
        if logging_dataset_id is None:
            logging_dataset_id = (
                dataset_id
                + "_"
                + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
                + "_pull_update_push"
            )

        with FileLogger(fn=f"{logging_dataset_id}.log", verbose=True):
            # Check collections and create completed list if needed
            collection_list = self.datasets.list()
            if logging_dataset_id not in collection_list["datasets"]:
                self.logger.info("Creating a logging collection for you.")
                self.logger.info(self.datasets.create(logging_dataset_id))

            # Track failed documents
            failed_documents: List[Dict] = []

            # Trust the process
            for _ in range(number_of_retrieve_retries):

                # Get document lengths to calculate iterations
                original_length = self.get_number_of_documents(dataset_id, filters)
                completed_length = self.get_number_of_documents(logging_dataset_id)
                remaining_length = original_length - completed_length
                iterations_required = math.ceil(remaining_length / retrieve_chunk_size)

                self.logger.debug(f"{original_length}")
                self.logger.debug(f"{completed_length}")
                self.logger.debug(f"{iterations_required}")

                # Return if no documents to update
                if remaining_length == 0:
                    self.logger.success(f"Pull, Update, Push is complete!")
                    return {
                        "failed_documents": failed_documents,
                        "logging_collection": logging_dataset_id,
                    }

                for _ in progress_bar(
                    range(iterations_required), show_progress_bar=show_progress_bar
                ):

                    # Get completed documents
                    log_json = self._get_all_documents(
                        logging_dataset_id, show_progress_bar=False
                    )
                    completed_documents_list = [i["_id"] for i in log_json]

                    # Get incomplete documents from raw collection
                    retrieve_filters = filters + [
                        {
                            "field": "ids",
                            "filter_type": "ids",
                            "condition": "!=",
                            "condition_value": completed_documents_list,
                        }
                    ]

                    orig_json = self.datasets.documents.get_where(
                        dataset_id,
                        filters=retrieve_filters,
                        page_size=retrieve_chunk_size,
                        select_fields=select_fields,
                    )

                    documents = orig_json["documents"]
                    self.logger.debug(f"{len(documents)}")

                    # Update documents
                    try:
                        updated_data = update_function(documents, **updating_args)
                    except Exception as e:
                        self.logger.error(
                            "Your updating function does not work: " + str(e)
                        )
                        traceback.print_exc()
                        return
                    updated_documents = [i["_id"] for i in documents]
                    self.logger.debug(f"{len(updated_data)}")

                    # Upload documents
                    if updated_dataset_id is None:
                        insert_json = self._update_documents(
                            dataset_id=dataset_id,
                            documents=updated_data,
                            max_workers=max_workers,
                            show_progress_bar=False,
                        )
                    else:
                        insert_json = self._insert_documents(
                            dataset_id=updated_dataset_id,
                            documents=updated_data,
                            max_workers=max_workers,
                            show_progress_bar=False,
                        )

                    # Check success
                    chunk_failed = insert_json["failed_documents"]
                    self.logger.success(
                        f"Chunk of {retrieve_chunk_size} original documents updated and uploaded with {len(chunk_failed)} failed documents!"
                    )
                    failed_documents.extend(chunk_failed)
                    success_documents = list(
                        set(updated_documents) - set(failed_documents)
                    )
                    upload_documents = [{"_id": i} for i in success_documents]

                    self._insert_documents(
                        logging_dataset_id,
                        upload_documents,
                        max_workers=max_workers,
                        show_progress_bar=False,
                    )

                    # If fail, try to reduce retrieve chunk
                    if len(chunk_failed) > 0:
                        warnings.warn(Warning.UPLOAD_FAILED)
                        retrieve_chunk_size = int(
                            retrieve_chunk_size
                            * retrieve_chunk_size_failure_retry_multiplier
                        )
                        time.sleep(self.config.seconds_between_retries)
                        break

                    if len(failed_documents) > max_error:
                        self.logger.error(
                            f"You have over {max_error} failed documents which failed to upload!"
                        )
                        return {
                            "failed_documents": failed_documents,
                            "logging_collection": logging_dataset_id,
                        }

                self.logger.success(f"Pull, Update, Push is complete!")

            return {
                "failed_documents": failed_documents,
                "logging_collection": logging_dataset_id,
            }

    @track
    def insert_df(self, dataset_id, dataframe, *args, **kwargs):
        """Insert a dataframe for each doc"""

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
            for doc in dataframe.to_dict(orient="records")
        ]
        results = self._insert_documents(dataset_id, documents, *args, **kwargs)
        self.print_search_dashboard_url(dataset_id)
        return results

    def _insert_csv(
        self,
        dataset_id: str,
        filepath_or_buffer,
        id_col: str = None,
        create_id: bool = True,
        max_workers: int = 2,
        retry_chunk_mult: float = 0.5,
        show_progress_bar: bool = False,
        csv_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        """
        Insert data from csv file

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        filepath_or_buffer :
            Any valid string path is acceptable. The string could be a URL. Valid URL schemes include http, ftp, s3, gs, and file.
        id_col : str
            Optional argument to use when a specific field is supposed to be used as the unique identifier ('_id')
        create_id: bool = True
            Automatically generateds UUID if create_id is True and if the '_id' field does not exist
        max_workers : int
            Number of workers active for multi-threading
        retry_chunk_mult: int
            Multiplier to apply to chunksize if upload fails

        Example
        ---------
        >>> from relevanceai import Client
        >>> client = Client()
        >>> df = client.Dataset("sample_dataset_id")
        >>> df.insert_csv("temp.csv")

        """
        df: pd.DataFrame = pd.read_csv(filepath_or_buffer, **csv_args)

        # Initialise output
        inserted = 0
        failed_documents = []
        failed_documents_detailed = []

        test_docs = self.json_encoder(df.loc[:19].to_dict("records"))
        doc_mbs = [getsizeof(test_doc) for test_doc in test_docs]
        doc_mb = sum(doc_mbs) / len(doc_mbs)
        doc_mb /= ONE_MB

        target_chunk_mb = int(self.config.get_option("upload.target_chunk_mb"))
        max_chunk_size = int(self.config.get_option("upload.max_chunk_size"))
        chunksize = (
            int(target_chunk_mb / doc_mb) + 1
            if int(target_chunk_mb / doc_mb) + 1 < df.shape[0]
            else df.shape[0]
        )
        chunksize = min(chunksize, max_chunk_size)

        # Add edge case handling
        if chunksize == 0:
            chunksize = 1

        # Chunk inserts
        for chunk in self.chunk(
            documents=df,
            chunksize=chunksize,
            show_progress_bar=show_progress_bar,
        ):
            response = self._insert_csv_chunk(
                chunk=chunk,
                dataset_id=dataset_id,
                id_col=id_col,
                create_id=create_id,
                max_workers=max_workers,
                retry_chunk_mult=retry_chunk_mult,
                show_progress_bar=False,
            )
            inserted += response["inserted"]
            failed_documents += response["failed_documents"]
            failed_documents_detailed += response["failed_documents_detailed"]

        return {
            "inserted": inserted,
            "failed_documents": failed_documents,
            "failed_documents_detailed": failed_documents_detailed,
        }

    def _insert_csv_chunk(
        self,
        chunk: pd.DataFrame,
        dataset_id: str,
        id_col: Optional[str] = None,
        create_id: bool = False,
        max_workers: int = 2,
        retry_chunk_mult: float = 0.5,
        show_progress_bar: bool = False,
    ):
        # generate '_id' if possible
        # id_col
        if "_id" not in chunk.columns and id_col:
            if id_col in chunk:
                chunk.insert(0, "_id", chunk[id_col], False)
            else:
                warnings.warn(Warning.COLUMN_DNE.format(id_col))
        # create_id
        if "_id" not in chunk.columns and create_id:
            index = chunk.index
            uuids = [
                make_id(chunk.iloc[chunk_index]) for chunk_index in range(len(index))
            ]
            chunk.insert(0, "_id", uuids, False)
            warnings.warn(Warning.AUTO_GENERATE_IDS)

        # Check for _id
        if "_id" not in chunk.columns:
            raise FieldNotFoundError("Need _id as a column")

        # add fix for when lists are read in as strings
        EXCEPTION_COLUMNS = ("_vector_", "_chunk_")
        vector_columns = [i for i in chunk.columns if i.endswith(EXCEPTION_COLUMNS)]
        for i in vector_columns:
            chunk[i] = chunk[i].apply(literal_eval)

        chunk_json = chunk.to_dict(orient="records")

        print(
            f"while inserting, you can visit your dashboard at https://cloud.relevance.ai/dataset/{dataset_id}/dashboard/monitor/"
        )
        response = self._insert_documents(
            dataset_id=dataset_id,
            documents=chunk_json,
            max_workers=max_workers,
            show_progress_bar=show_progress_bar,
        )
        return response

    def print_search_dashboard_url(self, dataset_id):
        search_url = (
            f"https://cloud.relevance.ai/dataset/{dataset_id}/deploy/recent/search"
        )
        self._dataset_id = dataset_id
        print(f"🍡 You can now explore your search app at {search_url}")

    def delete_pull_update_push_logs(self, dataset_id=False):

        collection_list = self.datasets.list()["datasets"]

        if dataset_id:
            log_collections = [
                i
                for i in collection_list
                if ("pull_update_push" in i) and (dataset_id in i)
            ]

        else:
            log_collections = [i for i in collection_list if ("pull_update_push" in i)]

        [self.datasets.delete(i, confirm=False) for i in log_collections]
        return

    def _write_documents(
        self,
        dataset_id: str,
        bulk_func: Callable,
        documents: List[Dict[str, Any]],
        max_workers: Optional[int] = 2,
        chunksize: Optional[int] = None,
        **kwargs,
    ):

        documents = self._validate_ids(documents)

        # Get one document to test the size
        if len(documents) == 0:
            warnings.warn(Warning.NO_DOCUMENT_DETECTED)
            return {
                "inserted": 0,
                "failed_documents": [],
                "failed_documents_detailed": [],
            }

        # Insert documents
        test_docs = self.json_encoder(documents[:20])
        doc_mbs = [getsizeof(test_doc) for test_doc in test_docs]
        doc_mb = sum(doc_mbs) / len(doc_mbs)
        doc_mb /= ONE_MB

        if chunksize is None:
            target_chunk_mb = int(self.config.get_option("upload.target_chunk_mb"))
            max_chunk_size = int(self.config.get_option("upload.max_chunk_size"))

            chunksize = math.ceil(target_chunk_mb / doc_mb)
            chunksize = min(chunksize, len(documents), max_chunk_size)

            payload_size = doc_mb * chunksize

            tqdm.write(
                f"Size (MB) / Document: {doc_mb:.3f}\nInsert Chunksize: {chunksize:,}\nPayload Size (MB): ~{payload_size:.2f}"
            )

        # handles if the client calls _insert_documents
        invert_op = getattr(self, "Dataset", None)
        if callable(invert_op):
            dataset = invert_op(dataset_id)
        else:
            dataset = self

        push = Push(
            dataset=dataset,
            func=bulk_func,
            func_kwargs=kwargs,
            documents=documents,
            chunksize=chunksize,
            max_workers=max_workers,
        )
        inserted, failed_ids = push.run()

        failed_ids_detailed: List[str] = []

        output = {
            "inserted": inserted,  # type: ignore
            "failed_documents": failed_ids,
            "failed_documents_detailed": failed_ids_detailed,
        }
        return output

    @beta
    def rename_fields(
        self,
        dataset_id: str,
        field_mappings: dict,
    ):
        """
        Loops through every document in your collection and renames specified fields by deleting the old one and
        creating a new field using the provided mapping
        These documents are then uploaded into either an updated collection, or back into the original collection.

        Example:
        rename_fields(dataset_id,field_mappings = {'a.b.d':'a.b.c'})  => doc['a']['b']['d'] => doc['a']['b']['c']
        rename_fields(dataset_id,field_mappings = {'a.b':'a.c'})  => doc['a']['b'] => doc['a']['c']

        Parameters
        ----------
        dataset_id : string
            The dataset_id of the collection where your original documents are
        field_mappings : dict
            A dictionary in the form f {old_field_name1 : new_field_name1, ...}
        retrieve_chunk_size: int
            The number of documents that are received from the original collection with each loop iteration.
        retrieve_chunk_size_failure_retry_multiplier: int
            If fails, retry on each chunk
        max_workers: int
            The number of processors you want to parallelize with
        show_progress_bar: bool
            Shows a progress bar if True
        """

        skip = set()
        for old_f, new_f in field_mappings.items():
            if len(old_f.split(".")) != len(new_f.split(".")):
                skip.add(old_f)
                warnings.warn(Warning.FIELD_MISMATCH.format(old_f, new_f))

        for k in skip:
            del field_mappings[k]

        def rename_dict_fields(d, field_mappings={}, track=""):
            modified_dict = {}
            for k, v in sorted(d.items(), key=lambda x: x[0]):
                if track != "" and track[-1] != ".":
                    track += "."
                if track + k not in field_mappings.keys():
                    kk = k
                else:
                    kk = field_mappings[track + k].split(".")[-1]

                if isinstance(v, dict):
                    if k not in modified_dict:
                        modified_dict[kk] = {}
                    modified_dict[kk] = rename_dict_fields(
                        v, field_mappings, track + kk
                    )
                else:
                    modified_dict[kk] = v
            return modified_dict

        sample_documents = self.datasets.documents.list(dataset_id)

        def update_function(sample_documents):
            for i, d in enumerate(sample_documents):
                sample_documents[i] = rename_dict_fields(
                    d, field_mappings=field_mappings, track=""
                )
            return sample_documents

        self.pull_update_push(dataset_id, update_function, retrieve_chunk_size=200)

    def _process_insert_results(self, results: dict, return_json: bool = False):
        # in case API is backwards incompatible

        if "failed_documents" in results:
            if len(results["failed_documents"]) == 0:
                tqdm.write(
                    ("✅ All documents inserted/edited successfully.")
                    .encode("utf-8")
                    .decode("utf8")
                )
            else:
                tqdm.write(
                    ("❗Few errors with inserting/editing documents. Please check logs.")
                    .encode("utf-8")
                    .decode("utf8")
                )
                return results

        elif "failed_document_ids" in results:
            if len(results["failed_document_ids"]) == 0:
                tqdm.write(
                    ("✅ All documents inserted/edited successfully.")
                    .encode("utf-8")
                    .decode("utf8")
                )
            else:
                tqdm.write(
                    ("❗Few errors with inserting/editing documents. Please check logs.")
                    .encode("utf-8")
                    .decode("utf8")
                )
                return results

        # Make backwards compatible on errors
        if (
            len(results.get("failed_documents", []))
            + len(results.get("failed_document_ids", []))
            > 0
        ) or return_json:
            return results

        return results
