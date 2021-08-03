"""All Dataset related functions
"""
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

    def schema(self, dataset_id: str, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/schema", method="GET", output_format = output_format, verbose = verbose)

    def metadata(self, dataset_id: str, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/metadata", method="GET", output_format = output_format, verbose = verbose)

    def stats(self, dataset_id: str, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/monitor/stats", method="GET", output_format = output_format, verbose = verbose)
    
    def health(self, dataset_id: str, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/monitor/health", method="GET", output_format = output_format, verbose = verbose)

    def create(self, dataset_id: str, schema: dict = {}, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint=f"datasets/create", method="POST",
                                    parameters={"id": dataset_id,
                                                "schema": schema},
                                    output_format = output_format, verbose = verbose)

    def list(self, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint="datasets/list", method="GET", output_format = output_format, verbose = verbose)

    def list_all(self, include_schema: bool = True, include_stats: bool = True, include_metadata: bool = True,
                        include_schema_stats: bool = False, include_vector_health: bool = False, include_active_jobs: bool = False, 
                        dataset_ids: list = [], sort_by_created_at_date: bool = False, asc: bool = False, page_size: int = 20, 
                        page: int = 1, output_format: str = "json", verbose: bool = True):
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
                output_format = output_format, verbose = verbose
                )
    
    def facets(self, dataset_id, fields: list = [], date_interval: str="monthly", 
        page_size: int=5, page: int=1, asc: bool=False, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint=f"datasets/{dataset_id}/facets",
            method="POST",
            parameters={
                "fields": fields,
                "date_interval": date_interval,
                "page_size": page_size,
                "page": page,
                "asc": asc
            }, output_format = output_format, verbose = verbose)

    def bulk_insert(self, dataset_id: str, documents: list, insert_date: bool = True, 
                    overwrite: bool = True, update_schema: bool = True, output_format: str = "json", verbose: bool = True):
        return self.make_http_request(
            # endpoint=f"datasets/{dataset_id}/documents/bulk_insert",
            endpoint=f"datasets/{dataset_id}/documents/bulk_insert",
            base_url="https://ingest-api-dev-aueast.relevance.ai/latest/",
            method="POST",
            parameters={
                "documents": documents,
                "insert_date": insert_date,
                "overwrite": overwrite,
                "update_schema": update_schema,
                # "include_inserted_ids": include_inserted_ids
            }, output_format=output_format, verbose = verbose)

    def delete(self, dataset_id: str, confirm = False, output_format: str = "json", verbose: bool = True):
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
            }, output_format = output_format, verbose = verbose
        )
        
        elif user_input.lower() in ('n', 'no'): 
            print(f'{dataset_id} not deleted')
            return 

        else:
           # ... error handling ...
           print(f'Error: Input {user_input} unrecognised.')
           return        

    def clone(self, old_dataset: str, new_dataset: str, schema: dict = {}, rename_fields: dict = {}, 
                        remove_fields: list = [], filters: list = [], output_format: str = "json", verbose: bool = True):
        return self.make_http_request(endpoint=f"datasets/{old_dataset}/clone", method="POST",
                                        parameters = {
                                                "new_dataset_id": new_dataset,
                                                "schema": schema,
                                                "rename_fields": rename_fields,
                                                "remove_fields": remove_fields,
                                                "filters": filters
                                            }, output_format = output_format, verbose = verbose)

    def get_number_of_documents(self, dataset_ids: list):
        collection_info = self.list_all(include_schema = False, include_stats = False, include_metadata = False,
                                        include_schema_stats = True, dataset_ids = dataset_ids, verbose = False)

        document_lengths = {i: collection_info['datasets'][i]['stats']['number_of_documents'] for i in dataset_ids}

        return document_lengths
    
    def search(self, query, sort_by_created_at_date: bool=False, asc: bool=False, 
        output_format: str="json", verbose: bool=True):
        return self.make_http_request(
            endpoint="datasets/search",
            method="POST",
            parameters={
                "query": query,
                "sort_by_created_at_date": sort_by_created_at_date,
                "asc": asc
            },
            output_format=output_format,
            verbose=verbose
        )

