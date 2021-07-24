"""Batch operations
"""
from tqdm import tqdm
from typing import Callable
from ..api.client import APIClient
from .chunk import Chunker
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from ..concurrency import multithread, multiprocess

class BatchInsert(APIClient, Chunker):
    def insert_documents(self, dataset_id: str, docs: list, 
        bulk_fn: Callable=None, verbose: bool=True,
        chunksize: int=10000, max_workers:int =8,  *args, **kwargs):
        """
        Insert a list of documents with multi-threading automatically
        enabled.
        """
        print(f"You are currently inserting into {dataset_id}")
        print(f"You can track your stats and progress via our dashboard at https://playground.getvectorai.com/collections/dashboard/stats/?collection={dataset_id}")
        def bulk_insert_func(docs):
            return self.datasets.bulk_insert(
                dataset_id,
                docs, verbose = verbose, *args, **kwargs)
        
        if bulk_fn is not None:
            return multiprocess(
                func=bulk_fn,
                iterables=docs,
                post_func_hook=bulk_insert_func,
                max_workers=max_workers,
                chunksize=chunksize)

        return multithread(bulk_insert_func, docs, 
            max_workers=max_workers, chunksize=chunksize)

    def pull_encode_push(self, raw_collection: str, completed_raw_collection:str, encoded_collection: str, 
                        encoding_function, encoding_args: dict = {}, retrieve_chunk_size: int = 100, 
                        upload_chunk_size: int = 1000, max_workers:int =8):

        #Check collections and create completed list if needed
        collection_list = self.datasets.list(verbose = 0)
        if completed_raw_collection not in collection_list:
            self.datasets.bulk_insert(completed_raw_collection, [{'_id': 'test'}], output_format = False, verbose = False)

        #Get document lengths
        collection_info = self.datasets.list_all(include_schema_stats = True, dataset_ids = [raw_collection, completed_raw_collection], verbose = False)
        raw_length = collection_info['datasets'][raw_collection]['schema_stats']['insert_date_']['missing'] + collection_info['datasets'][raw_collection]['schema_stats']['insert_date_']['exists']
        completed_length = collection_info['datasets'][completed_raw_collection]['schema_stats']['insert_date_']['missing'] + collection_info['datasets'][completed_raw_collection]['schema_stats']['insert_date_']['exists']

        remaining_length = raw_length - completed_length
        iterations_required =  int(remaining_length/retrieve_chunk_size) + 1

        
        #Trust the process
        for i in tqdm(range(iterations_required)):

            #Get completed documents
            x = self.datasets.documents.get_where_all(completed_raw_collection, verbose = False)
            completed_documents_list = [i['_id'] for i in x]

            #Get incomplete documents from raw collection
            y = self.datasets.documents.get_where(raw_collection, 
                                                    filters = [
                                                    {"field": "ids", "filter_type": "ids", "condition": "!=", "condition_value": completed_documents_list}
                                                    ],
                                                    page_size = retrieve_chunk_size, verbose = False)

            documents = y['documents']

            try:                                          
                encoded_data = encoding_function(documents, **encoding_args)
            except Exception as e:
                print('Your encoding function does not work: ' + e)
                return

            encoded_documents = [{'_id': i['_id']} for i in documents]

            z = self.insert_documents(dataset_id = encoded_collection, docs = encoded_data, chunksize = upload_chunk_size, output_format = False, max_workers = max_workers)

            if 200 in set([i.status_code for i in z]) and len(set([i.status_code for i in z])) == 1:
                self.insert_documents(completed_raw_collection, encoded_documents, chunksize = 10000, output_format = False, max_workers = max_workers)
                print('Chunk encoded and uploaded!')
            
            else:
                print('Chunk FAILED to encode and upload!')
                return {"Failed Chunk": encoded_documents}
