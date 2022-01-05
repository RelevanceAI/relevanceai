"""
Migrate from mongo database to Relevance Ai:
    #Create an object of Mongo2RelevanceAi class
    connection_string= "..."
    project= "..."
    api_key= "..."
    mongo2vec = Mongo2Mongo2RelevanceAi(connection_string, project, api_key)

    #Get a summary of the mondo database using "mongo_summary"
    mongo2vec.mongo_summary()

    #Set the desired source mongo collection using "set_mongo_collection"
    db_name = '...'
    collection_name = '...'
    mongo2vec.set_mongo_collection(db_name, collection_name)

    #Get total number of entries in the mongo collection using "mongo_doc_count"
    doc_cnt = mongo2vec.mongo_doc_count()

    #Migrate data from mongo to Relevance Ai using "migrate_mongo2relevance_ai"
    chunk_size = 5000      # migrate batches of 5000 (default 2000)
    start_idx= 12000       # loads from mongo starting at index 12000 (default 0)
    dataset_id = "..."     # dataset id in the Relevance Ai platform
    mongo2vec.migrate_mongo2relevance_ai(dataset_id, doc_cnt, chunk_size = chunk_size, start_idx= start_idx)
"""

import copy
import json
import math
import numpy as np
import pandas as pd
import uuid
import warnings
from tqdm.auto import tqdm
from typing import List
from relevanceai.api.client import BatchAPIClient

try:
    from pymongo import MongoClient

    PYMONGO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYMONGO_AVAILABLE = False
    warnings.warn(
        "you are missing `pymongo.MongoClient`. Please install this using `pip install pymongo==3.12`"
    )

try:
    from bson import json_util

    BSON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BSON_AVAILABLE = False
    warnings.warn(
        "you are missing `bson.json_util`. Please install this using `pip install bson`"
    )


class Mongo2RelevanceAi(BatchAPIClient):
    def __init__(self, connection_string: str, project: str, api_key: str):
        super().__init__(project, api_key)
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

    def mongo_doc_count(self):
        return self.mongo_collection.count()

    def create_relevance_ai_dataset(self, dataset_id: str):
        response = self.datasets.create(dataset_id)
        return response

    def update_id(self, docs: List[dict]):
        # makes bson id format json campatible
        for doc in docs:
            try:
                doc["_id"] = doc["_id"]["$oid"]
            except Exception as e:
                self.logger.info("Could not use the original id: " + str(e))
                doc["_id"] = uuid.uuid4().__str__()
        return docs

    @staticmethod
    def parse_json(data):
        return json.loads(json_util.dumps(data))

    @staticmethod
    def flatten_inner_indxs(docs: List[dict]):
        # {f1:{f2:v}} => {f1-f2:v}
        expanded = copy.deepcopy(docs)
        for i, doc in enumerate(docs):
            for f, v in doc.items():
                if isinstance(v, dict):
                    del expanded[i][f]
                    for k in v:
                        expanded[i][f + "-" + k] = v[k]
        return expanded

    @staticmethod
    def remove_nan(docs: List[dict], replace_with: str = ""):
        for doc in docs:
            for f, v in doc.items():
                if isinstance(v, float) and math.isnan(v) or v == np.NaN:
                    doc[f] = replace_with
        return docs

    @staticmethod
    def build_range(doc_cnt: int, chunk_size: int = 2000, start_idx: int = 0):
        rng = [
            (s, s + chunk_size)
            if s + chunk_size <= start_idx + doc_cnt
            else (s, start_idx + doc_cnt)
            for s in list(range(start_idx, start_idx + doc_cnt, chunk_size))
        ]
        return rng

    def fetch_mongo_collection_data(self, start_idx: int = None, end_idx: int = None):
        if start_idx and end_idx:
            return list(self.mongo_collection.find()[start_idx:end_idx])
        return list(self.mongo_collection.find())

    def migrate_mongo2relevance_ai(
        self,
        dataset_id: str,
        doc_cnt: int,
        chunk_size: int = 2000,
        start_idx: int = 0,
        overwite: bool = False,
    ):
        response = self.create_relevance_ai_dataset(dataset_id)
        if "already exists" in response["message"] and not overwite:
            self.logger.error(response["message"])
            return response["message"]

        total_ingest_cnt = 0
        for s_idx, e_idx in tqdm(
            Mongo2RelevanceAi.build_range(doc_cnt, chunk_size, start_idx)
        ):
            df = pd.DataFrame(self.fetch_mongo_collection_data(s_idx, e_idx))
            docs = self.update_id(Mongo2RelevanceAi.parse_json(df.to_dict("records")))
            docs = Mongo2RelevanceAi.remove_nan(
                Mongo2RelevanceAi.flatten_inner_indxs(docs)
            )
            self.insert_documents(dataset_id, docs)
            total_ingest_cnt += len(docs)

        self.logger.info(
            f"Successfully ingested {total_ingest_cnt} entities to {dataset_id}."
        )
        return f"Successfully ingested {total_ingest_cnt} entities to {dataset_id}."
