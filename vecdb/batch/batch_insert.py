"""Batch operations
"""
import json
import math
import sys
import time
import traceback
from datetime import datetime
from typing import Callable

from ..api.client import APIClient
from ..concurrency import multiprocess, multithread
from ..progress_bar import progress_bar
from .chunk import Chunker

BYTE_TO_MB = 1024 * 1024
LIST_SIZE_MULTIPLIER = 3


class BatchInsert(APIClient, Chunker):
    def insert_documents(
        self,
        dataset_id: str,
        docs: list,
        bulk_fn: Callable = None,
        verbose: bool = True,
        max_workers: int = 8,
        retry_chunk_mult: int = 0.5,
        show_progress_bar: bool = False,
        chunksize=100,
        max_retries=1,
        *args,
        **kwargs,
    ):

        """
        Insert a list of documents with multi-threading automatically
        enabled.
        """
        if verbose:
            self.logger.info(f"You are currently inserting into {dataset_id}")
        if verbose:
            self.logger.info(
                f"You can track your stats and progress via our dashboard at https://cloud.relevance.ai/collections/dashboard/stats/?collection={dataset_id}"
            )
        # Check if the collection exists
        self.datasets.create(dataset_id, output_format=None, verbose=False)

        def bulk_insert_func(docs):
            return self.datasets.bulk_insert(
                dataset_id,
                docs,
                verbose=verbose,
                return_documents=True,
                retries=max_retries,
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
        verbose: bool = True,
        max_workers: int = 8,
        retry_chunk_mult: int = 0.5,
        chunksize: int = 100,
        show_progress_bar=False,
        *args,
        **kwargs,
    ):
        """
        Update a list of documents with multi-threading
        automatically enabled.
        This is useful especially when pull_update_push bugs out and you need somethin urgently.
        Set chunksize to `None` for automatic conversion

        >>> from vecdb import VecDBClient
        >>>  url = "https://api-aueast.relevance.ai/v1/"

        >>> collection = ""
        >>> project = ""
        >>> api_key = ""
        >>> client = VecDBClient(project, api_key)
        >>> docs = client.datasets.documents.get_where(collection, select_fields=['title'])
        >>> while len(docs['documents']) > 0:
        >>>     docs['documents'] = model.encode_documents_in_bulk(['product_name'], docs['documents'])
        >>>     client.update_documents(collection, docs['documents'])
        >>>     docs = client.datasets.documents.get_where(collection, select_fields=['product_name'], cursor=docs['cursor'])

        """
        if verbose:
            self.logger.info(f"You are currently updating {dataset_id}")
        if verbose:
            self.logger.info(
                f"You can track your stats and progress via our dashboard at https://cloud.relevance.ai/collections/dashboard/stats/?collection={dataset_id}"
            )

        def bulk_update_func(docs):
            return self.datasets.documents.bulk_update(
                dataset_id,
                docs,
                verbose=verbose,
                return_documents=True,
                retries=1,
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
        logging_collection: str = None,
        updating_args: dict = {},
        retrieve_chunk_size: int = 100,
        retrieve_chunk_size_failure_retry_multiplier: float = 0.5,
        number_of_retrieve_retries: int = 3,
        max_workers: int = 8,
        max_error: int = 1000,
        filters: list = [],
        select_fields: list = [],
        verbose: bool = True,
        show_progress_bar: bool = True,
    ):
        """
        Loops through every document in your collection and applies a function (that is specified by you) to the documents. These documents are then uploaded into either an updated collection, or back into the original collection.

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
        if logging_collection == None:
            now = datetime.now()
            dt_string = now.strftime("_log_update_started_%d-%m-%Y_%H-%M-%S")
            logging_collection = original_collection + dt_string

        # Check collections and create completed list if needed
        collection_list = self.datasets.list(verbose=False)
        if logging_collection not in collection_list["datasets"]:
            self.logger.info("Creating a logging collection for you.")
            self.logger.info(
                self.datasets.create(
                    logging_collection, output_format="json", verbose=verbose
                )
            )

        # Track failed documents
        failed_documents = []

        # Trust the process

        for _ in range(number_of_retrieve_retries):

            # Get document lengths to calculate iterations
            original_length = self.datasets.documents._get_number_of_documents(
                original_collection, filters
            )
            completed_length = self.datasets.documents._get_number_of_documents(
                logging_collection
            )
            remaining_length = original_length - completed_length
            iterations_required = math.ceil(remaining_length / retrieve_chunk_size)

            self.logger.debug(f"{original_length}")
            self.logger.debug(f"{completed_length}")
            self.logger.debug(f"{iterations_required}")

            # Return if no documents to update
            if remaining_length == 0:
                self.logger.success(f"Pull, Update, Push is complete!")
                return {
                    "Failed Documents": failed_documents,
                    "Logging Collection": logging_collection,
                }

            for _ in progress_bar(
                range(iterations_required), show_progress_bar=show_progress_bar
            ):

                # Get completed documents
                log_json = self.datasets.documents.get_where_all(
                    logging_collection, verbose=verbose
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
                    original_collection,
                    filters=retrieve_filters,
                    page_size=retrieve_chunk_size,
                    select_fields=select_fields,
                    verbose=verbose,
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
                        verbose=verbose,
                        max_workers=max_workers,
                        show_progress_bar=False,
                    )
                else:
                    insert_json = self.insert_documents(
                        dataset_id=updated_collection,
                        docs=updated_data,
                        verbose=verbose,
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
                    verbose=False,
                    max_workers=max_workers,
                )

                # If fail, try to reduce retrieve chunk
                if len(chunk_failed) > 0:
                    self.logger.warning(
                        "Failed to upload. Retrieving half of previous number."
                    )
                    retrieve_chunk_size = (
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
                        "Failed Documents": failed_documents,
                        "Logging Collection": logging_collection,
                    }

        self.logger.success(f"Pull, Update, Push is complete!")
        return {
            "Failed Documents": failed_documents,
            "Logging Collection": logging_collection,
        }

    def insert_df(self, dataset_id, dataframe, *args, **kwargs):
        """Insert a dataframe for eachd doc"""
        import pandas as pd

        docs = [
            {k: v for k, v in doc.items() if not pd.isna(v)}
            for doc in dataframe.to_dict(orient="records")
        ]
        return self.insert_documents(dataset_id, docs, *args, **kwargs)

    def delete_all_logs(self, dataset_id):
        collection_list = self.datasets.list()["datasets"]
        log_collections = [
            i
            for i in collection_list
            if ("log_update_started" in i) and (dataset_id in i)
        ]
        [self.datasets.delete(i, confirm=False) for i in log_collections]
        return

    def _write_documents(
        self,
        insert_function,
        docs: list,
        bulk_fn: Callable = None,
        max_workers: int = 8,
        retry_chunk_mult: int = 0.5,
        show_progress_bar: bool = False,
        chunksize: int = None,
    ):

        # Get one document to test the size
        test_doc = json.dumps(docs[0], indent=4)
        doc_mb = sys.getsizeof(test_doc) * LIST_SIZE_MULTIPLIER / BYTE_TO_MB
        if chunksize is None:
            chunksize = (
                int(self.config.get_option("upload.target_chunk_mb") / doc_mb)
                if int(self.config.get_option("upload.target_chunk_mb") / doc_mb)
                < len(docs)
                else len(docs)
            )

        # Initialise number of inserted documents
        inserted = []

        # Initialise failed documents
        failed_ids = [i["_id"] for i in docs]

        # Initialise failed documents detailed
        failed_ids_detailed = []

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
                [
                    inserted.append(chunk["response_json"]["inserted"])
                    for chunk in insert_json
                    if chunk["status_code"] == 200
                ]
                for chunk in insert_json:

                    # Track failed in 200
                    if chunk["status_code"] == 200:
                        [
                            failed_ids.append(i["_id"])
                            for i in chunk["response_json"]["failed_documents"]
                        ]

                        [
                            failed_ids_detailed.append(i)
                            for i in chunk["response_json"]["failed_documents"]
                        ]

                    # Cancel documents with 400 or 404
                    elif chunk["status_code"] in [400, 404]:
                        [cancelled_ids.append(i["_id"]) for i in chunk["documents"]]

                    # Half chunksize with 413 or 524
                    elif chunk["status_code"] in [413, 524]:
                        [failed_ids.append(i["_id"]) for i in chunk["documents"]]
                        chunksize = chunksize * retry_chunk_mult

                    # Retry all other errors
                    else:
                        [failed_ids.append(i["_id"]) for i in chunk["documents"]]

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
