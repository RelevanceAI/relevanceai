"""
Migrate from mongo database to vecdb:
    #Create an object of Mongo2Vecbd class
    connection_string= "..."
    project= "..."
    api_key= "..."
    mongo2vec = Mongo2Vecbd(connection_string, project, api_key)

    #Get a summary of the mondo db using "mongo_summary"
    mongo2vec.mongo_summary()

    #Set the desired source mongo collection using "set_mongo_collection"
    db_name = '...'
    collection_name = '...'
    mongo2vec.set_mongo_collection(db_name, collection_name)

    #Get total number of entries in the mongo collection using "mongo_doc_count"
    doc_cnt = mongo2vec.mongo_doc_count()

    #Migrate data from mongo to vecdb using "migrate_mongo2vecdb"
    stp = 5000             # migrate batches of 5000 (default 2000)
    start_idx= 12000       # loads from mongo starting at index 12000 (default 0)
    vecdb_collection_name = "..."
    mongo2vec.migrate_mongo2vecdb(vecdb_collection_name, doc_cnt, stp = stp, start_idx= start_idx)
"""

from ..api.client import APIClient
from vecdb import VecDBClient
from pymongo import MongoClient
import pandas as pd
import numpy as np
import json
from bson import json_util
import math
from tqdm import tqdm
import copy

class Mongo2Vecbd(APIClient):
    def __init__(self, connection_string:str, project:str, api_key:str, base_url = "https://api-aueast.relevance.ai/v1/"):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        self.vecdb_client = VecDBClient(project, api_key, base_url = self.base_url)
        self.vecdb_credential = project + ":" + api_key
        self.mongo_client = MongoClient(connection_string)
    
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
    
    def get_mongo_db(self, db_name):
        return self.mongo_client[db_name]

    def get_mongo_collection(self, db_name, collection_name):
        return self.mongo_client[db_name][collection_name]

    def set_mongo_db(self, db_name):
        self.mongodb = self.mongo_client[db_name]

    def set_mongo_collection(self, db_name, collection_name):
        self.mongo_collection = self.mongo_client[db_name][collection_name]

    def mongo_doc_count(self):
      return self.mongo_collection.count()

    def create_vcdb_collection(self, collection_name):
        self.vecdb_client.datasets.create(collection_name)

    @staticmethod
    def parse_json(data):
        return json.loads(json_util.dumps(data))

    @staticmethod
    def update_id(docs):
        # makes bson id format json campatible
        for doc in docs:
            doc['_id'] = doc['_id']['$oid']
        return docs

    @staticmethod
    def flatten_innder_indxs(docs):
        # {f1:{f2:v}} => {f1-f2:v}
        expanded = copy.deepcopy(docs)
        for i,doc in enumerate(docs):
            for f,v in doc.items():
                if isinstance(v,dict):
                    del expanded[i][f]
                    for k in v:
                        expanded[i][f+'-'+k] = v[k]
        return expanded

    @staticmethod
    def remove_nan(docs, replace_with = ""):
        for doc in docs:
            for f,v in doc.items():
                if isinstance(v, float) and math.isnan(v) or v == np.NaN:
                    doc[f]=replace_with
        return docs

    @staticmethod
    def build_range(doc_cnt, stp, start = 0):
        rng = [(s, s+stp) if s+stp < doc_cnt 
               else (s, doc_cnt) 
               for s in list(range(start, doc_cnt, stp))]
        return rng

    def fetch_mongo_collection_data(self, rng = None):
        if rng:
          s,e = rng
          return list(self.mongo_collection.find()[s:e])
        return list(self.mongo_collection.find())

    def migrate_mongo2vecdb(self, vecdb_collection, doc_cnt, stp = 2000, start_idx = 0, create_new = True):
        # todo: check if it doesn't exist, should we remove an existing one?
        self.create_vcdb_collection(vecdb_collection)

        for rng in tqdm(Mongo2Vecbd.build_range(doc_cnt, stp, start_idx)): 
            df = pd.DataFrame(self.fetch_mongo_collection_data(rng))
            docs = Mongo2Vecbd.update_id(Mongo2Vecbd.parse_json(df.to_dict('records')))
            docs = Mongo2Vecbd.remove_nan(Mongo2Vecbd.flatten_innder_indxs(docs))
            self.vecdb_client.datasets.bulk_insert(vecdb_collection, docs)