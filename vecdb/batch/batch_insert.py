"""Batch operations
"""
from ..progress_bar import progress_bar
from typing import Callable
from ..api.client import APIClient
from .chunk import Chunker
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from ..concurrency import multithread, multiprocess
import traceback
from datetime import datetime

class BatchInsert(APIClient, Chunker):
    def insert_documents(self, dataset_id: str, docs: list, 
        bulk_fn: Callable=None, verbose: bool=True,
        chunksize: int=10000, max_workers:int =8,  *args, **kwargs):
        """
        Insert a list of documents with multi-threading automatically
        enabled.
        """
        if verbose: print(f"You are currently inserting into {dataset_id}") 
        if verbose: print(f"You can track your stats and progress via our dashboard at https://playground.getvectorai.com/collections/dashboard/stats/?collection={dataset_id}") 
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

    def update_documents(self, dataset_id: str, docs: list, 
        bulk_fn: Callable=None, verbose: bool=True,
        chunksize: int=10000, max_workers:int =8,  *args, **kwargs):
        """
        Update a list of documents with multi-threading
        automatically enabled.
        """
        if verbose: print(f"You are currently updating {dataset_id}") 
        if verbose: print(f"You can track your stats and progress via our dashboard at https://playground.getvectorai.com/collections/dashboard/stats/?collection={dataset_id}") 
        def bulk_update_func(docs):
            return self.datasets.documents.bulk_update(
                dataset_id,
                docs, verbose = verbose, *args, **kwargs)
        
        if bulk_fn is not None:
            return multiprocess(
                func=bulk_fn,
                iterables=docs,
                post_func_hook=bulk_update_func,
                max_workers=max_workers,
                chunksize=chunksize)

        return multithread(bulk_update_func, docs, 
            max_workers=max_workers, chunksize=chunksize)



    def pull_update_push(self, 
        original_collection: str, update_function, 
        updated_collection: str = None, 
        logging_collection:str = None,
        updating_args: dict = {}, 
        retrieve_chunk_size: int = 100, 
        upload_chunk_size: int = 1000, max_workers:int =8, max_error: int = 1000, 
        select_fields: list=[],
        verbose: bool=True):

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

        upload_chunk_size: int
            The number of documents that are uploaded with each loop iteration.

        max_workers: int
            The number of processors you want to parallelize with

        max_error: 
            How many failed uploads before the function breaks

        """

        #Check if a logging_collection has been supplied
        if logging_collection == None:
            now = datetime.now()
            dt_string = now.strftime("_log_update_started_%d-%m-%Y_%H-%M-%S")
            logging_collection = original_collection + dt_string

        #Check collections and create completed list if needed
        collection_list = self.datasets.list(verbose = False)
        if logging_collection not in collection_list['datasets']:
            print("Creating a logging collection for you.")
            print(self.datasets.create(logging_collection, output_format = 'json', verbose = verbose))

        #Get document lengths to calculate iterations
        collection_lengths = self.datasets.get_number_of_documents([original_collection, logging_collection])
        original_length = collection_lengths[original_collection]
        completed_length = collection_lengths[logging_collection]
        remaining_length = original_length - completed_length
        iterations_required =  int(remaining_length/retrieve_chunk_size) + 1

        #Track failed documents
        failed_documents = []

        #Trust the process
        for i in progress_bar(range(iterations_required)):

            #Get completed documents
            x = self.datasets.documents.get_where_all(logging_collection, verbose = verbose)
            completed_documents_list = [i['_id'] for i in x]

            #Get incomplete documents from raw collection
            y = self.datasets.documents.get_where(
                original_collection, 
                filters = [
                    {"field": "ids", "filter_type": "ids", "condition": "!=", "condition_value": completed_documents_list}
                ],
                page_size = retrieve_chunk_size, 
                select_fields=select_fields,
                verbose = verbose)

            documents = y['documents']

            #Update documents
            try:                                          
                updated_data = update_function(documents, **updating_args)
            except Exception as e:
                print('Your updating function does not work: ' + str(e))
                traceback.print_exc()
                return
            updated_documents = [i['_id'] for i in documents]

            #Upload documents   
            if updated_collection is None: 
                z = self.update_documents(dataset_id = original_collection, docs = updated_data, verbose = verbose, 
                    chunksize = upload_chunk_size, max_workers = max_workers)
            else:
                z = self.insert_documents(dataset_id = updated_collection, docs = updated_data, 
                    verbose = verbose, chunksize = upload_chunk_size, max_workers = max_workers)

            #Check success
            chunk_failed = []
            check = [chunk_failed.extend(i['failed_documents']) for i in z if i is not None]
            print(f'Chunk of {retrieve_chunk_size} original documents updated and uploaded with {len(chunk_failed)} failed documents!')
            failed_documents.extend(chunk_failed)

            success_documents = list(set(updated_documents) - set(failed_documents))
            upload_documents = [{'_id': i} for i in success_documents]
            self.insert_documents(logging_collection, upload_documents, verbose = False, chunksize = 10000, max_workers = max_workers)

            if len(failed_documents) > max_error:
                print(f'You have over {max_error} failed documents which failed to upload!')
                return {"Failed Documents": failed_documents}

        return


    def pull_update_push_v2(self, 
        original_collection: str, update_function, 
        logging_field:str = None, 
        updating_args: dict = {}, 
        retrieve_chunk_size: int = 100, 
        upload_chunk_size: int = 1000, max_workers:int =8, max_error: int = 1000, 
        select_fields: list=[],
        verbose: bool=True):

        """
        Loops through every document in your collection and applies a function (that is specified to you) to the documents. These documents are then uploaded into either an updated collection, or back into the original collection. 

        Parameters
        ----------
        original_collection : string
            The dataset_id of the collection where your original documents are

        logging_field: string
            The field of the collection which logs which documents have been updated. If 'None', then one will be created for you based on the date and time. This field can be reused if documents are not fully updated.

        update_function: function
            A function created by you that converts documents in your original collection into the updated documents. The function must input a list of documents (the original collection) and output another list of documents (to be updated).
        updating_args: dict
            Additional arguments to your update_function, if they exist. They must be in the format of {'Argument': Value}

        retrieve_chunk_size: int
            The number of documents that are received from the original collection with each loop iteration.

        upload_chunk_size: int
            The number of documents that are uploaded with each loop iteration.

        max_workers: int
            The number of processors you want to parallelize with

        max_error: 
            How many failed uploads before the function breaks

        """

        #Check if a logging_field has been supplied, otherwise use the current date
        if logging_field == None:
            now = datetime.now()
            dt_string = now.strftime("Log Update Started (%d/%m/%Y, %H:%M:%S)")
            logging_field = dt_string

        #Get logging_field length to calculate iterations

        dataset_fields = self.datasets.health(original_collection)
        if logging_field in dataset_fields:
            remaining_length = dataset_fields[logging_field]['missing']
        else:
            remaining_length = self.datasets.get_number_of_documents([original_collection])[original_collection]

        iterations_required =  int(remaining_length/retrieve_chunk_size) + 1

        #Track failed documents
        failed_documents = []

        #Trust the process
        for i in progress_bar(range(iterations_required)):

            #Get incomplete documents from original collection
            y = self.datasets.documents.get_where(
                original_collection, 
                filters = [
                     {'field' : logging_field, 'filter_type' : 'exists', "condition":"!=", "condition_value":" "}
                ],
                page_size = retrieve_chunk_size, 
                select_fields=select_fields,
                verbose = verbose)
            documents = y['documents']

            #Update documents
            try:                                          
                updated_data = update_function(documents, **updating_args)
            except Exception as e:
                print('Your updating function does not work: ' + e)
                traceback.print_exc()
                return

            #Upload documents   
            updated_logged_data = [dict(i, **{logging_field:'Updated'}) for i in updated_data]
            z = self.update_documents(dataset_id = original_collection, docs = updated_logged_data, verbose = verbose, 
                chunksize = upload_chunk_size, max_workers = max_workers)

            #Check success
            chunk_failed = []
            check = [chunk_failed.extend(i['failed_documents']) for i in z if i is not None]
            print(f'Chunk of {retrieve_chunk_size} original documents updated and uploaded with {len(chunk_failed)} failed documents!')
            failed_documents.extend(chunk_failed)

            if len(failed_documents) > max_error:
                print(f'You have over {max_error} failed documents which failed to upload!')
                return {"Failed Documents": failed_documents}

        return
    
    def insert_df(self, dataset_id, dataframe, *args, **kwargs):
        """Insert a dataframe for eachd doc"""
        import pandas as pd
        docs = [{k:v for k, v in doc.items() if not pd.isna(v)} for doc in \
            dataframe.to_dict(orient='records')]
        return self.insert_documents(dataset_id, docs, *args, **kwargs)
