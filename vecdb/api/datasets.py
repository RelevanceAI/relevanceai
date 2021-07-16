"""All Dataset related functions
"""
import pandas as pd
from typing import Callable
from tqdm import tqdm
from ..base import Base
from .tasks import Tasks
from .documents import Documents
from .monitor import Monitor

class Datasets(Base):
    """All dataset-related functions
    """
    def __init__(self, project: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.project = project
        self.api_key = api_key
        self.tasks = Tasks(project=project, api_key=api_key, 
            base_url=base_url)
        self.documents = Documents(project=project, api_key=api_key,
            base_url=base_url)
        self.monitor = Monitor(project=project, api_key=api_key,
            base_url=base_url)

    def get_where(self, dataset_id: str, filters: list=[], cursor: str=None, 
        page_size: int=20, sort: list=[], select_fields: list=[], 
        include_vector: bool=True, random_state: int = 0, is_random: bool = False, output_format: str = "json"):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/get_where", 
            method="POST", 
            parameters={
                "select_fields": select_fields,
                "cursor": cursor,
                "page_size": page_size,
                "sort": sort,
                "include_vector": include_vector,
                "filters": filters,
                "random_state": random_state,
                "is_random": is_random}
            , output_format = output_format)
    
    def schema(self, dataset_id: str, output_format: str = "json"):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/schema", method="GET", output_format = output_format)

    def metadata(self, dataset_id: str, output_format: str = "json"):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/metadata", method="GET", output_format = output_format)

    def list(self, output_format: str = "json"):
        return self.make_http_request(endpoint="datasets/list", method="GET", output_format = output_format)

    def list_all(self, include_schema: bool = True, include_stats: bool = True, include_metadata: bool = True,
                        include_schema_stats: bool = False, include_vector_health: bool = False, include_active_jobs: bool = False, 
                        dataset_ids: list = [], sort_by_created_at_date: bool = False, asc: bool = False, page_size: int = 20, 
                        page: int = 1, output_format: str = "json"):
        return self.make_http_request(endpoint="datasets/list", 
                method="POST",
                parameters={
                "include_schema": include_schema,
                "include_stats": include_stats,
                "include_metadata": include_metadata,
                "include_schema_stats": include_schema_stats,
                "include_vector_health": include_vector_health,
                "include_active_jobs": include_active_jobs,
                "dataset_ids": dataset_ids,
                "sort_by_created_at_date": sort_by_created_at_date,
                "asc": asc,
                "page_size": page_size,
                "page": page},
                output_format = output_format
                )
    
    def facets(self, dataset_id, fields: list = [], date_interval: str="monthly", 
        page_size: int=5, page: int=1, asc: bool=False, output_format: str = "json"):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/facets",
            method="POST",
            parameters={
                "fields": fields,
                "date_interval": date_interval,
                "page_size": page_size,
                "page": page,
                "asc": asc
            }, output_format = output_format)

    def bulk_insert(self, dataset_id: str, documents: list, insert_date: bool = True, 
                    overwrite: bool = True, update_schema: bool = True, include_inserted_ids: bool = False, output_format: str = "json"):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/documents/bulk_insert",
            method="POST",
            parameters={
                "documents": documents,
                "insert_date": insert_date,
                "overwrite": overwrite,
                "update_schema": update_schema,
                "include_inserted_ids": include_inserted_ids
            }, output_format = output_format)

    def delete(self, dataset_id: str, confirm = True, output_format: str = "json"):
        if confirm == True:
            # confirm with the user
            print(f'You are about to delete {dataset_id}')
            user_input = input('Confirm? [Y/N] ')
        else: 
            user_input = 'y'
        # input validation
        if user_input.lower() in ('y', 'yes'):
            return self.make_http_request(
            endpoint=f"datasets/delete",
            method='POST',
            parameters={
                "dataset_id": dataset_id
            }, output_format = output_format
        )
        
        elif user_input.lower() in ('n', 'no'): 
            print(f'{dataset_id} not deleted')
            return 

        else:
           # ... error handling ...
           print(f'Error: Input {user_input} unrecognised.')
           return        
        
    def get_where_all(self, dataset_id: str, chunk_size: int = 10000, filters: list=[], sort: list=[], select_fields: list=[], include_vector: bool=True, random_state: int = 0, output_format: str = "json"):
        #Initialise values
        length = 1
        cursor = None
        full_data = []

        #While there is still data to fetch, fetch it at the latest cursor
        while length > 0 :
            x = self.get_where(dataset_id, filters=filters, cursor=cursor, page_size= chunk_size, sort = sort, select_fields = select_fields, include_vector = include_vector, random_state = random_state, output_format = output_format)
            length = len(x['documents'])
            cursor = x['cursor']

            #Append fetched data to the full data
            if length > 0:
                [full_data.append(i) for i in x['documents']]
        
        return full_data

    def chunk(self, docs: list, chunksize: int=15):
        for i in range(int(len(docs) / chunksize) + 1):
            yield docs[i*chunksize: chunksize*(i+1)]


    def bulk_insert_chunk(self, dataset_id: str, documents: list, chunksize: int = 15, 
                            insert_date: bool = True, overwrite: bool = True, update_schema: bool = True, include_inserted_ids: bool = False, output_format: str = "json"):
        for i in tqdm(self.chunk(docs = documents, chunksize = chunksize)):
            self.bulk_insert(dataset_id = dataset_id, 
                            documents = i, 
                            insert_date = insert_date, 
                            overwrite = overwrite, 
                            update_schema = update_schema, 
                            include_inserted_ids = include_inserted_ids, 
                            output_format = output_format)

        return

    def copy_collections(self, old_dataset: str, new_dataset: str, schema: dict = {}, rename_fields: dict = {}, remove_fields: list = [], filters: list = [], output_format: str = "json"):
        return self.make_http_request(endpoint=f"datasets/{old_dataset}/clone", method="POST",
                                        parameters = {
                                                "new_dataset_id": new_dataset,
                                                "schema": schema,
                                                "rename_fields": rename_fields,
                                                "remove_fields": remove_fields,
                                                "filters": filters
                                            }, output_format = output_format)

