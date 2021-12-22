"""Batch Insert"""
import json
import math
import sys
import time
import traceback
from datetime import datetime
from typing import Callable, List, Dict, Union, Any

from relevanceai.api.endpoints.client import APIClient
from relevanceai.api.batch.batch_retrieve import BatchRetrieveClient
from relevanceai.api.batch.local_logger import PullUpdatePushLocalLogger
from relevanceai.concurrency import multiprocess, multithread
from relevanceai.progress_bar import progress_bar
from relevanceai.api.batch.chunk import Chunker

BYTE_TO_MB = 1024 * 1024
LIST_SIZE_MULTIPLIER = 3


class BatchInsertClient(BatchRetrieveClient, APIClient, Chunker):
    def insert_documents(
        self,
        dataset_id: str,
        docs: list,
        bulk_fn: Callable = None,
        max_workers: int = 8,
        retry_chunk_mult: float = 0.5,
        show_progress_bar: bool = False,
        chunksize=0,
        *args,
        **kwargs,
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
        docs : list
            A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
        bulk_fn : callable
            Function to apply to documents before uploading
        max_workers : int
            Number of workers active for multi-threading
        retry_chunk_mult: int
            Multiplier to apply to chunksize if upload fails
        chunksize : int
            Number of documents to upload per worker. If None, it will default to the size specified in config.upload.target_chunk_mb
        """

        self.logger.info(f"You are currently inserting into {dataset_id}")

        self.logger.info(
            f"You can track your stats and progress via our dashboard at https://cloud.relevance.ai/collections/dashboard/stats/?collection={dataset_id}"
        )
        # Check if the collection exists
        self.datasets.create(dataset_id)

        def bulk_insert_func(docs):
            return self.datasets.bulk_insert(
                dataset_id,
                docs,
                return_documents=True,
                *args,
                **kwargs,
            )

        return self._write_documents(
            bulk_insert_func,
            docs,
            bulk_fn,
            max_workers,
            retry_chunk_mult,
            show_progress_bar=show_progress_bar,
            chunksize=chunksize,
        )

    def update_documents(
        self,
        dataset_id: str,
        docs: list,
        bulk_fn: Callable = None,
        max_workers: int = 8,
        retry_chunk_mult: float = 0.5,
        chunksize: int = 0,
        show_progress_bar=False,
        *args,
        **kwargs,
    ):
        """
        Update a list of documents with multi-threading automatically enabled.
        Edits documents by providing a key value pair of fields you are adding or changing, make sure to include the "_id" in the documents.

        >>> from relevanceai import Client
        >>> url = "https://api-aueast.relevance.ai/v1/"
        >>> collection = ""
        >>> project = ""
        >>> api_key = ""
        >>> client = Client(project, api_key)
        >>> docs = client.datasets.documents.get_where(collection, select_fields=['title'])
        >>> while len(docs['documents']) > 0:
        >>>     docs['documents'] = model.encode_documents_in_bulk(['product_name'], docs['documents'])
        >>>     client.update_documents(collection, docs['documents'])
        >>>     docs = client.datasets.documents.get_where(collection, select_fields=['product_name'], cursor=docs['cursor'])

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        docs : list
            A list of documents. Document is a JSON-like data that we store our metadata and vectors with. For specifying id of the document use the field '_id', for specifying vector field use the suffix of '_vector_'
        bulk_fn : callable
            Function to apply to documents before uploading
        max_workers : int
            Number of workers active for multi-threading
        retry_chunk_mult: int
            Multiplier to apply to chunksize if upload fails
        chunksize : int
            Number of documents to upload per worker. If None, it will default to the size specified in config.upload.target_chunk_mb
        """

        self.logger.info(f"You are currently updating {dataset_id}")

        self.logger.info(
            f"You can track your stats and progress via our dashboard at https://cloud.relevance.ai/collections/dashboard/stats/?collection={dataset_id}"
        )

        def bulk_update_func(docs):
            return self.datasets.documents.bulk_update(
                dataset_id,
                docs,
                return_documents=True,
                *args,
                **kwargs,
            )

        return self._write_documents(
            bulk_update_func,
            docs,
            bulk_fn,
            max_workers,
            retry_chunk_mult,
            chunksize=chunksize,
        )

    def pull_update_push(
        self,
        original_collection: str,
        update_function,
        updated_collection: str = None,
        log_file: str = None,
        updating_args: dict = {},
        retrieve_chunk_size: int = 100,
        max_workers: int = 8,
        filters: list = [],
        select_fields: list = [],
        show_progress_bar: bool = True,
    ):
        """
        Loops through every document in your collection and applies a function (that is specified by you) to the documents.
        These documents are then uploaded into either an updated collection, or back into the original collection.

        Parameters
        ----------
        original_collection : string
            The dataset_id of the collection where your original documents are
        logging_collection: string
            The dataset_id of the collection which logs which documents have been updated. If 'None', then one will be created for you.
        updated_collection: string
            The dataset_id of the collection where your updated documents are uploaded into. If 'None', then your original collection will be updated.
        update_function: function
            A function created by you that converts documents in your original collection into the updated documents. The function must contain a field which takes in a list of documents from the original collection. The output of the function must be a list of updated documents.
        updating_args: dict
            Additional arguments to your update_function, if they exist. They must be in the format of {'Argument': Value}
        retrieve_chunk_size: int
            The number of documents that are received from the original collection with each loop iteration.
        max_workers: int
            The number of processors you want to parallelize with
        max_error:
            How many failed uploads before the function breaks
        """
        if not callable(update_function):
            raise TypeError(
                "Your update function needs to be a function! Please read the documentation if it is not."
            )

        # Check if a logging_collection has been supplied
        if log_file is None:
            log_file = (
                original_collection
                + "_"
                + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
                + "_pull_update_push"
                + ".log"
            )

        # Instantiate the logger to document the successful IDs
        PULL_UPDATE_PUSH_LOGGER = PullUpdatePushLocalLogger(log_file)

        # Track failed documents
        failed_documents: List[Dict] = []

        # Trust the process
        # Get document lengths to calculate iterations
        original_length = self.get_number_of_documents(original_collection, filters)

        # get the remaining number in case things break
        remaining_length = original_length - PULL_UPDATE_PUSH_LOGGER.count_ids_in_fn()
        iterations_required = math.ceil(remaining_length / retrieve_chunk_size)

        completed_documents_list: list = []

        # Get incomplete documents from raw collection
        retrieve_filters = filters + [
            {
                "field": "ids",
                "filter_type": "ids",
                "condition": "!=",
                "condition_value": completed_documents_list,
            }
        ]

        for _ in progress_bar(
            range(iterations_required), show_progress_bar=show_progress_bar
        ):

            orig_json = self.datasets.documents.get_where(
                original_collection,
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
            if updated_collection is None:
                insert_json = self.update_documents(
                    dataset_id=original_collection,
                    docs=updated_data,
                    max_workers=max_workers,
                    show_progress_bar=False,
                )
            else:
                insert_json = self.insert_documents(
                    dataset_id=updated_collection,
                    docs=updated_data,
                    max_workers=max_workers,
                    show_progress_bar=False,
                )

            # Check success
            chunk_failed = insert_json["failed_documents"]
            failed_documents.extend(chunk_failed)
            success_documents = list(set(updated_documents) - set(failed_documents))
            PULL_UPDATE_PUSH_LOGGER.log_ids(success_documents)
            self.logger.success(
                f"Chunk of {retrieve_chunk_size} original documents updated and uploaded with {len(chunk_failed)} failed documents!"
            )

        self.logger.success(f"Pull, Update, Push is complete!")

        # if PULL_UPDATE_PUSH_LOGGER.count_ids_in_fn() == original_length:
        #     os.remove(log_file)
        return {
            "failed_documents": failed_documents,
        }

    def pull_update_push_to_cloud(
        self,
        original_collection: str,
        update_function,
        updated_collection: str = None,
        logging_collection: str = None,
        updating_args: dict = {},
        retrieve_chunk_size: int = 100,
        retrieve_chunk_size_failure_retry_multiplier: float = 0.5,
        number_of_retrieve_retries: int = 3,
        max_workers: int = 8,
        max_error: int = 1000,
        filters: list = [],
        select_fields: list = [],
        show_progress_bar: bool = True,
    ):
        """
        Loops through every document in your collection and applies a function (that is specified by you) to the documents.
        These documents are then uploaded into either an updated collection, or back into the original collection.

        Parameters
        ----------
        original_collection : string
            The dataset_id of the collection where your original documents are
        logging_collection: string
            The dataset_id of the collection which logs which documents have been updated. If 'None', then one will be created for you.
        updated_collection: string
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
        max_error:
            How many failed uploads before the function breaks
        """
        # Check if a logging_collection has been supplied
        if logging_collection is None:
            logging_collection = (
                original_collection
                + "_"
                + str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
                + "_pull_update_push"
            )

        # Check collections and create completed list if needed
        collection_list = self.datasets.list()
        if logging_collection not in collection_list["datasets"]:
            self.logger.info("Creating a logging collection for you.")
            self.logger.info(self.datasets.create(logging_collection))

        # Track failed documents
        failed_documents: List[Dict] = []

        # Trust the process
        for _ in range(number_of_retrieve_retries):

            # Get document lengths to calculate iterations
            original_length = self.get_number_of_documents(original_collection, filters)
            completed_length = self.get_number_of_documents(logging_collection)
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
                    "logging_collection": logging_collection,
                }

            for _ in progress_bar(
                range(iterations_required), show_progress_bar=show_progress_bar
            ):

                # Get completed documents
                log_json = self.get_all_documents(logging_collection)
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
                    original_collection,
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
                    self.logger.error("Your updating function does not work: " + str(e))
                    traceback.print_exc()
                    return
                updated_documents = [i["_id"] for i in documents]
                self.logger.debug(f"{len(updated_data)}")

                # Upload documents
                if updated_collection is None:
                    insert_json = self.update_documents(
                        dataset_id=original_collection,
                        docs=updated_data,
                        max_workers=max_workers,
                        show_progress_bar=False,
                    )
                else:
                    insert_json = self.insert_documents(
                        dataset_id=updated_collection,
                        docs=updated_data,
                        max_workers=max_workers,
                        show_progress_bar=False,
                    )

                # Check success
                chunk_failed = insert_json["failed_documents"]
                self.logger.success(
                    f"Chunk of {retrieve_chunk_size} original documents updated and uploaded with {len(chunk_failed)} failed documents!"
                )
                failed_documents.extend(chunk_failed)
                success_documents = list(set(updated_documents) - set(failed_documents))
                upload_documents = [{"_id": i} for i in success_documents]

                self.insert_documents(
                    logging_collection,
                    upload_documents,
                    max_workers=max_workers,
                )

                # If fail, try to reduce retrieve chunk
                if len(chunk_failed) > 0:
                    self.logger.warning(
                        "Failed to upload. Retrieving half of previous number."
                    )
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
                        "logging_collection": logging_collection,
                    }

        self.logger.success(f"Pull, Update, Push is complete!")
        return {
            "failed_documents": failed_documents,
            "logging_collection": logging_collection,
        }

    def insert_df(self, dataset_id, dataframe, *args, **kwargs):
        """Insert a dataframe for eachd doc"""
        import pandas as pd

        docs = [
            {k: v for k, v in doc.items() if not pd.isna(v)}
            for doc in dataframe.to_dict(orient="records")
        ]
        return self.insert_documents(dataset_id, docs, *args, **kwargs)

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
        insert_function,
        docs: list,
        bulk_fn: Callable = None,
        max_workers: int = 8,
        retry_chunk_mult: float = 0.5,
        show_progress_bar: bool = False,
        chunksize: int = 0,
    ):

        # Get one document to test the size
        if len(docs) == 0:
            self.logger.warning("No document is detected")
            return {
                "inserted": 0,
                "failed_documents": [],
                "failed_documents_detailed": [],
            }

        # Insert documents
        test_doc = json.dumps(docs[0], indent=4)
        doc_mb = sys.getsizeof(test_doc) * LIST_SIZE_MULTIPLIER / BYTE_TO_MB
        if chunksize == 0:
            target_chunk_mb = int(self.config.get_option("upload.target_chunk_mb"))
            max_chunk_size = int(self.config.get_option("upload.max_chunk_size"))
            chunksize = (
                int(target_chunk_mb / doc_mb) + 1
                if int(target_chunk_mb / doc_mb) + 1 < len(docs)
                else len(docs)
            )
            chunksize = max(chunksize, max_chunk_size)

        # Initialise number of inserted documents
        inserted: List[str] = []

        # Initialise failed documents
        failed_ids = [i["_id"] for i in docs]

        # Initialise failed documents detailed
        failed_ids_detailed: List[str] = []

        # Initialise cancelled documents
        cancelled_ids = []

        for i in range(int(self.config.get_option("retries.number_of_retries"))):
            if len(failed_ids) > 0:
                if bulk_fn is not None:
                    insert_json = multiprocess(
                        func=bulk_fn,
                        iterables=docs,
                        post_func_hook=insert_function,
                        max_workers=max_workers,
                        chunksize=chunksize,
                        show_progress_bar=show_progress_bar,
                    )
                else:
                    insert_json = multithread(
                        insert_function,
                        docs,
                        max_workers=max_workers,
                        chunksize=chunksize,
                        show_progress_bar=show_progress_bar,
                    )

                failed_ids = []
                failed_ids_detailed = []

                # Update inserted amount
                def is_successfully_inserted(chunk: Union[Dict, Any]) -> bool:
                    return chunk["status_code"] == 200

                inserted += list(
                    map(
                        lambda x: x["response_json"]["inserted"],
                        filter(is_successfully_inserted, insert_json),
                    )
                )

                for chunk in insert_json:

                    # Track failed in 200
                    if chunk["status_code"] == 200:
                        failed_ids += [
                            i["_id"] for i in chunk["response_json"]["failed_documents"]
                        ]

                        failed_ids_detailed += [
                            i for i in chunk["response_json"]["failed_documents"]
                        ]

                    # Cancel documents with 400 or 404
                    elif chunk["status_code"] in [400, 404]:
                        cancelled_ids += [i["_id"] for i in chunk["documents"]]

                    # Half chunksize with 413 or 524
                    elif chunk["status_code"] in [413, 524]:
                        failed_ids += [i["_id"] for i in chunk["documents"]]
                        chunksize = int(chunksize * retry_chunk_mult)

                    # Retry all other errors
                    else:
                        failed_ids += [i["_id"] for i in chunk["documents"]]

                # Update docs to retry which have failed
                docs = [i for i in docs if i["_id"] in failed_ids]

            else:
                break

        # When returning, add in the cancelled ids
        failed_ids.extend(cancelled_ids)

        output = {
            "inserted": sum(inserted),
            "failed_documents": failed_ids,
            "failed_documents_detailed": failed_ids_detailed,
        }
        return output
