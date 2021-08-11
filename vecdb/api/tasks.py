"""Tasks Module
"""
from ..base import Base
import time

class Tasks(Base):
    def __init__(self, project: str, api_key: str, base_url: str):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url

    def create(self, dataset_id, task_name, task_parameters):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": task_name,
                **task_parameters
            }
        )

    def status(self, dataset_id: str, task_id: str):
        return self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/{task_id}/status",
            method="GET", verbose = False)

    def loop_status_until_finish(self, dataset_id: str, task_id: str, verbose: bool = True, time_between_ping: int = 10):
            status = False

            while status not in ['Finished', 'success']:
                time.sleep(time_between_ping)
                try:
                    status = self.status(dataset_id, task_id)['status']
                except:
                    print(f'Status-check timed out: {task_id}')
                    return task_id

                if verbose == True:
                    print(status)

            print(f"Your task is {status}!")
            return 


    def check_status_until_finish(self, dataset_id: str, task_id: str, status_checker: bool = True, verbose: bool = True, time_between_ping: int = 10):

        if status_checker == True:
            print(f"Task_ID: {task_id}")
            self.loop_status_until_finish(dataset_id, task_id, verbose = verbose, time_between_ping = time_between_ping)
            return 

        else:
            print("To view the progress of your job, visit https://cloud.relevanceai.com/collections/dashboard/jobs")
            return {"task_id": task_id}

    
    # Note: The following tasks are instantiated manually to accelerate 
    # creation of certain popular tasks

    #Make decorator wrat for all task checkers

    def create_cluster_task(self, dataset_id, vector_field: str, 
        n_clusters: int, alias: str="default", refresh: bool=False,
        n_iter: int=10, n_init: int=5, status_checker: bool = True, verbose: bool = True, time_between_ping: int = 10):
        task = self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": "Clusterer",
                "vector_field": vector_field,
                "n_clusters": n_clusters,
                "alias": alias,
                "refresh": refresh,
                "n_iter": n_iter,
                "n_init": n_init
                }
            )

        output = self.check_status_until_finish(dataset_id, task['task_id'], status_checker = status_checker, verbose = verbose, time_between_ping = time_between_ping)
        return output

    def create_numeric_encoder_task(self, dataset_id: str, fields: list, status_checker: bool = True, verbose: bool = True, time_between_ping: int = 10):
        """
        Within a collection encode the specified dictionary field in every document into vectors.\n
        For example: a dictionary that represents a **person's characteristics visiting a store, field "person_characteristics"**:\n
            document 1 field: {"person_characteristics" : {"height":180, "age":40, "weight":70}}\n
            document 2 field: {"person_characteristics" : {"age":32, "purchases":10, "visits": 24}}\n
            -> <Encode the dictionaries to vectors> ->\n
        | height | age | weight | purchases | visits |
        |--------|-----|--------|-----------|--------|
        | 180    | 40  | 70     | 0         | 0      |
        | 0      | 32  | 0      | 10        | 24     |\n
            document 1 dictionary vector: {"person_characteristics_vector_": [180, 40, 70, 0, 0]}\n
            document 2 dictionary vector: {"person_characteristics_vector_": [0, 32, 0, 10, 24]}
            """
        task = self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": "NumericEncoder",
                "fields": fields
            }
        )

        output = self.check_status_until_finish(dataset_id, task['task_id'], status_checker = status_checker, verbose = verbose, time_between_ping = time_between_ping)
        return output

    def create_encode_categories_task(self, dataset_id: str, fields: list, status_checker: bool = True, verbose: bool = True, time_between_ping: int = 10):
        """Within a collection encode the specified array field in every document into vectors.\n
        For example, array that represents a ****movie's categories, field "movie_categories"**:\n
            document 1 array field: {"category" : ["sci-fi", "thriller", "comedy"]}\n
            document 2 array field: {"category" : ["sci-fi", "romance", "drama"]}\n
            -> <Encode the arrays to vectors> ->\n
        | sci-fi | thriller | comedy | romance | drama |
        |--------|----------|--------|---------|-------|
        | 1      | 1        | 1      | 0       | 0     |
        | 1      | 0        | 0      | 1       | 1     |\n
            document 1 array vector: {"movie_categories_vector_": [1, 1, 1, 0, 0]}\n
            document 2 array vector: {"movie_categories_vector_": [1, 0, 0, 1, 1]}
        """
        task = self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": "CategoriesEncoder",
                "fields": fields
            }
        )

        output = self.check_status_until_finish(dataset_id, task['task_id'], status_checker = status_checker, verbose = verbose, time_between_ping = time_between_ping)
        return output

    
    def create_encode_text_task(self, dataset_id: str, field: str, alias: str="default", refresh: bool=False, status_checker: bool = True, verbose: bool = True, time_between_ping: int = 10):
        task = self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": "TextEncoder",
                "dataset_id": dataset_id,
                "field": field,
                "alias": alias,
                "refresh": refresh
            }
        )

        output = self.check_status_until_finish(dataset_id, task['task_id'], status_checker = status_checker, verbose = verbose, time_between_ping = time_between_ping)
        return output
    
    def create_encode_textimage_task(self, dataset_id: str, field: str, alias: str="default", refresh: bool=False, status_checker: bool = True, verbose: bool = True, time_between_ping: int = 10):
        task = self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": "TextImageEncoder",
                "dataset_id": dataset_id,
                "field": field,
                "alias": alias,
                "refresh": refresh
            }
        )

        output = self.check_status_until_finish(dataset_id, task['task_id'], status_checker = status_checker, verbose = verbose, time_between_ping = time_between_ping)
        return output
    
    def create_encode_imagetext_task(self, dataset_id: str, field: str, alias: str="default", refresh: bool=False, status_checker: bool = True, verbose: bool = True, time_between_ping: int = 10):
        task = self.make_http_request(
            endpoint=f"datasets/{dataset_id}/tasks/create",
            method='POST',
            parameters={
                "task_name": "ImageTextEncoder",
                "dataset_id": dataset_id,
                "field": field,
                "alias": alias,
                "refresh": refresh
            }
        )

        output = self.check_status_until_finish(dataset_id, task['task_id'], status_checker = status_checker, verbose = verbose, time_between_ping = time_between_ping)
        return output
