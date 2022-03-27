"""
Migrate from mongo database to Relevance Ai:

.. code-block::

    from relevanceai.api.batch import MongoImporter

    # Create an object of MongoImporter class
    connection_string= "..."
    project= "..."
    api_key= "..."
    mongo_importer = MongoImporter(connection_string, project, api_key)

    # Get a summary of the mondo database using "mongo_summary"
    mongo_importer.mongo_summary()

    # Set the desired source mongo collection using "set_mongo_collection"
    db_name = '...'
    collection_name = '...'
    mongo_importer.set_mongo_collection(db_name, dataset_id)

    # Get total number of entries in the mongo collection using "mongo_document_count"
    document_count = mongo_importer.mongo_document_count()

    # Migrate data from mongo to Relevance AI using "migrate_mongo2relevance_ai"
    chunk_size = 5000      # migrate batches of 5000 (default 2000)
    start_idx= 12000       # loads from mongo starting at index 12000 (default 0)
    dataset_id = "..."     # dataset id in the Relevance Ai platform
    mongo_importer.migrate(
        dataset_id, document_count, chunk_size=chunk_size,
        start_idx=start_idx)

"""

import copy
import json
import math
import numpy as np
import pandas as pd
import warnings

from tqdm.auto import tqdm
from typing import List

from relevanceai.utils import make_id
from relevanceai.constants.warning import Warning

try:
    from relevanceai import Client
    from pymongo import MongoClient

    PYMONGO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYMONGO_AVAILABLE = False
    warnings.warn(Warning.MISSING_PACKAGE)

try:
    from bson import json_util

    BSON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BSON_AVAILABLE = False
    warnings.warn(Warning.MISSING_PACKAGE)


class MongoImporter(Client):
    def __init__(self, connection_string: str):
        super().__init__()
        if PYMONGO_AVAILABLE:
            self.mongo_client = MongoClient(connection_string)
        else:
            self.logger.error(
                "you are missing Pymongo. Please install this using `pip install pymongo==3.12`"
            )

    def mongo_summary(self):
        """
        returns a dictionary {key:value}
        key = db names
        value = collection names in each db
        """
        summary = {}
        for db in self.mongo_client.database_names():
            summary[db] = []
            for collection in self.mongo_client[db].collection_names():
                summary[db].append(collection)
        return summary

    def get_mongo_db(self, db_name: str):
        return self.mongo_client[db_name]

    def get_mongo_collection(self, db_name: str, collection_name: str):
        return self.mongo_client[db_name][collection_name]

    def set_mongo_db(self, db_name: str):
        self.mongodb = self.mongo_client[db_name]

    def set_mongo_collection(self, db_name: str, collection_name: str):
        self.mongo_collection = self.mongo_client[db_name][collection_name]

    def mongo_document_count(self):
        return self.mongo_collection.count()

    def create_relevance_ai_dataset(self, dataset_id: str):
        response = self.datasets.create(dataset_id)
        return response

    def update_id(self, documents: List[dict]):
        # makes bson id format json campatible
        for document in documents:
            try:
                document["_id"] = document["_id"]["$oid"]
            except Exception as e:
                self.logger.info("Could not use the original id: " + str(e))
                document["_id"] = make_id(document)
        return documents

    @staticmethod
    def parse_json(data):
        return json.loads(json_util.dumps(data))

    @staticmethod
    def flatten_inner_indxs(documents: List[dict]):
        # {f1:{f2:v}} => {f1-f2:v}
        expanded = copy.deepcopy(documents)
        for i, doc in enumerate(documents):
            for f, v in doc.items():
                if isinstance(v, dict):
                    del expanded[i][f]
                    for k in v:
                        expanded[i][f + "-" + k] = v[k]
        return expanded

    @staticmethod
    def remove_nan(documents: List[dict], replace_with: str = ""):
        for doc in documents:
            for f, v in doc.items():
                if isinstance(v, float) and math.isnan(v) or v == np.NaN:
                    doc[f] = replace_with
        return documents

    @staticmethod
    def build_range(document_count: int, chunk_size: int = 2000, start_idx: int = 0):
        rng = [
            (s, s + chunk_size)
            if s + chunk_size <= start_idx + document_count
            else (s, start_idx + document_count)
            for s in list(range(start_idx, start_idx + document_count, chunk_size))
        ]
        return rng

    def fetch_mongo_collection_data(self, start_idx: int = None, end_idx: int = None):
        if start_idx and end_idx:
            return list(self.mongo_collection.find()[start_idx:end_idx])
        return list(self.mongo_collection.find())

    def migrate(
        self,
        dataset_id: str,
        document_count: int,
        chunk_size: int = 2000,
        start_idx: int = 0,
        overwite: bool = False,
    ):
        """
        Migrate your MongoDB dataset ID.

        Parameters
        ------------
        dataset_id: str
            Name of your dataset
        document_count: int
            The number of documents in your collection
        chunk_size: int
            The number of chunks
        start_idx: int
            The start index in case it breaks
        overwrite: bool
            If True, then the dataset ID in Relevance AI will be overwritten

        """
        response = self.create_relevance_ai_dataset(dataset_id)
        if "already exists" in response["message"] and not overwite:
            self.logger.error(response["message"])
            return response["message"]

        total_ingest_cnt = 0
        for s_idx, e_idx in tqdm(
            MongoImporter.build_range(document_count, chunk_size, start_idx)
        ):
            df = pd.DataFrame(self.fetch_mongo_collection_data(s_idx, e_idx))
            documents = self.update_id(MongoImporter.parse_json(df.to_dict("records")))
            documents = MongoImporter.remove_nan(
                MongoImporter.flatten_inner_indxs(documents)
            )
            self.insert_documents(dataset_id, documents)
            total_ingest_cnt += len(documents)

        self.logger.info(
            f"Successfully ingested {total_ingest_cnt} entities to {dataset_id}."
        )
        return f"Successfully ingested {total_ingest_cnt} entities to {dataset_id}."
