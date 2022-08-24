# -*- coding: utf-8 -*-
"""Batch Insert"""
import os
import sys
import json
import math
import uuid
import time

import warnings
import traceback

import numpy as np
import pandas as pd

from ast import literal_eval

from datetime import datetime

from typing import Any, Callable, Dict, List, Optional, Union

from tqdm.auto import tqdm

from relevanceai._api.batch.retrieve import BatchRetrieveClient
from relevanceai._api.batch.local_logger import PullUpdatePushLocalLogger

from tqdm.auto import tqdm

from relevanceai.utils import make_id
from relevanceai.utils.helpers.helpers import getsizeof
from relevanceai.utils.logger import FileLogger
from relevanceai.utils.progress_bar import progress_bar
from relevanceai.utils.decorators.version import beta
from relevanceai.utils.decorators.analytics import track
from relevanceai.utils.concurrency import Push, multiprocess, multithread

from relevanceai.constants.errors import FieldNotFoundError
from relevanceai.constants.warning import Warning
from relevanceai.constants import (
    MB_TO_BYTE,
    LIST_SIZE_MULTIPLIER,
    SUCCESS_CODES,
    RETRY_CODES,
    HALF_CHUNK_CODES,
)


class BatchInsertClient(BatchRetrieveClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _insert_documents(
        self,
        dataset_id: str,
        documents: list,
        max_workers: Optional[int] = 2,
        show_progress_bar: bool = False,
        batch_size: Optional[int] = None,
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
            batch_size=batch_size,
            overwrite=overwrite,
            ingest_in_background=ingest_in_background,
        )

    def _update_documents(
        self,
        dataset_id: str,
        documents: List[Dict[str, Any]],
        max_workers: Optional[int] = 2,
        show_progress_bar: bool = False,
        batch_size: Optional[int] = None,
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
            batch_size=batch_size,
            ingest_in_background=ingest_in_background,
        )

    update_documents = _update_documents

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
        df = pd.read_csv(filepath_or_buffer, **csv_args)

        # Initialise output
        inserted = 0
        failed_documents = []
        failed_documents_detailed = []

        test_doc = json.dumps(self.json_encoder(df.loc[0].to_dict()))
        doc_mb = sys.getsizeof(test_doc) * LIST_SIZE_MULTIPLIER / MB_TO_BYTE

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
        batch_size: Optional[int] = None,
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
        test_doc = self.json_encoder(documents[0])
        doc_mb = getsizeof(test_doc) * LIST_SIZE_MULTIPLIER / MB_TO_BYTE

        if batch_size is None:
            target_chunk_mb = int(self.config.get_option("upload.target_chunk_mb"))
            max_chunk_size = int(self.config.get_option("upload.max_chunk_size"))

            batch_size = math.ceil(target_chunk_mb / doc_mb)
            batch_size = min(batch_size, len(documents), max_chunk_size)

            tqdm.write(f"Updating chunksize for batch data insertion to {batch_size}")
            # Add edge case handling
            if batch_size == 0:
                batch_size = 1

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
            batch_size=batch_size,
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
                tqdm.write("✅ All documents inserted/edited successfully.")
            else:
                tqdm.write(
                    "❗Few errors with inserting/editing documents. Please check logs."
                )
                return results

        elif "failed_document_ids" in results:
            if len(results["failed_document_ids"]) == 0:
                tqdm.write("✅ All documents inserted/edited successfully.")
            else:
                tqdm.write(
                    "❗Few errors with inserting/editing documents. Please check logs."
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
