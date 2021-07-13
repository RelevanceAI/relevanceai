"""All Dataset related functions
"""
from .base import Base
from tqdm import tqdm

class Datasets(Base):
    """All dataset-related functions
    """
    def __init__(self, project: str, api_key: str, base_url: str):
        self.base_url = base_url
        self.project = project
        self.api_key = api_key
    
    def list(self, dataset_id: str):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/documents/list", method="GET")

    def get_where(self, dataset_id: str, filters: list=[], cursor: str=None, page_size: int=20, 
        sort: list=[], select_fields: list=[], include_vector: bool=True):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/documents/get_where", 
            method="POST", 
            parameters={
                "select_fields": select_fields,
                "cursor": cursor,
                "page_size": page_size,
                "sort": sort,
                "include_vector": include_vector,
                "filters": filters})
    
    def schema(self, dataset_id: str):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/schema", method="GET")

    def list_dataset(self):
        return self.make_http_request(endpoint=f"datasets/list", method="GET")

    def create_task(self, dataset_id, task_name, task_parameters):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": task_name,
                **task_parameters
            }
        )
    
    def facets(self, dataset_id, fields: list, date_interval: str="monthly", 
        page_size: int=5, page: int=1, asc: bool=False):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/facets",
            method="POST",
            parameters={
                "fields": fields,
                "date_interval": date_interval,
                "page_size": page_size,
                "page": page,
                "asc": asc
            })


    def bulk_insert(self, dataset_id: str, documents: list, insert_date: bool = True, overwrite: bool = True, update_schema: bool = True, include_inserted_ids: bool = False, output_json: bool = True):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/documents/bulk_insert",
            method="POST",
            parameters={
                "documents": documents,
                "insert_date": insert_date,
                "overwrite": overwrite,
                "update_schema": update_schema,
                "include_inserted_ids": include_inserted_ids
            }, output_json = output_json)

    def delete(self, dataset_id: str, confirm = True):

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
            }
        )
        
        elif user_input.lower() in ('n', 'no'): 
            print(f'{dataset_id} not deleted')
            return 

        else:
           # ... error handling ...
           print(f'Error: Input {user_input} unrecognised.')
           return
        
        
    def get_where_all(self, dataset_id: str, chunk_size: int = 10000, filters: list=[], sort: list=[], select_fields: list=[], include_vector: bool=True):

        #Initialise values
        length = 1
        cursor = None
        full_data = []

        #While there is still data to fetch, fetch it at the latest cursor
        while length > 0 :
            x = self.get_where(dataset_id, filters=filters, cursor=cursor, page_size= chunk_size, sort = sort, select_fields = select_fields, include_vector = include_vector)
            length = len(x['documents'])
            cursor = x['cursor']

            #Append fetched data to the full data
            if length > 0:
                [full_data.append(i) for i in x['documents']]
        
        return full_data


    def chunk(self, docs: list, chunksize: int=15):
        for i in range(int(len(docs) / chunksize) + 1):
            yield docs[i*chunksize: chunksize*(i+1)]


    def bulk_insert_chunk(self, dataset_id: str, documents: list, chunksize: int = 15, insert_date: bool = True, overwrite: bool = True, update_schema: bool = True, include_inserted_ids: bool = False, output_json: bool = True):
        for i in tqdm(self.chunk(docs = documents, chunksize = chunksize)):
            self.bulk_insert(dataset_id = dataset_id, 
                            documents = i, 
                            insert_date = insert_date, 
                            overwrite = overwrite, 
                            update_schema = update_schema, 
                            include_inserted_ids = include_inserted_ids, 
                            output_json = output_json)

        return
