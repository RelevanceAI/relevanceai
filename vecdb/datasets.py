"""All Dataset related functions
"""
from .base import Base

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


    def upload_documents(self, dataset_id: str, data: list, insert_date = True, overwrite = True, update_schema = True, include_inserted_ids = False):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/documents/bulk_insert",
            method="POST",
            parameters={
                "documents": data,
                "insert_date": insert_date,
                "overwrite": overwrite,
                "update_schema": update_schema,
                "include_inserted_ids": include_inserted_ids
            })


    def create_dataset(self, dataset_id: str, data: list, upload = False):
        self.make_http_request(
                endpoint=f"datasets/create",
                method='POST',
                parameters={
                    "id": dataset_id
                }
            )

        if upload == False:
            return
            
        else:
            self.upload_documents(dataset_id, data)
            return 


    def delete_dataset(self, dataset_id: str):
        # confirm with the user

        print(f'You are about to delete {dataset_id}')
        user_input = input('Confirm? [Y/N] ')

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
        
        


