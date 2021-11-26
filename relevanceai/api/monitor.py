"""All Dataset related functions
"""
from relevanceai.base import Base


class Monitor(Base):
    def __init__(self, project, api_key, base_url):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def health(
        self, dataset_id: str, output_format: str = "json", verbose: bool = True
    ):
        """ 
        Gives you a summary of the health of your vectors, e.g. how many documents with vectors are missing, how many documents with zero vectors 
        
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/monitor/health",
            method="GET",
            output_format=output_format,
            verbose=verbose,
        )


    def stats(self, dataset_id: str, output_format: str = "json", verbose: bool = True):
        """ 
        All operations related to monitoring
        
        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/monitor/stats",
            method="GET",
            output_format=output_format,
            verbose=verbose,
        )

    def usage(self, dataset_id: str, filters: list = [], page_size: int = 20, page: int = 1, asc: bool = False, flatten: bool = True, log_ids: list = [], output_format: str = "json", verbose: bool = True):
        """ 
        Aggregate the logs for a dataset. \n

        The response returned has the following fields:

        >>> [{'frequency': 958, 'insert_date': 1630159200000},...]

        Parameters
        ----------
        dataset_id: string
            Unique name of dataset
        filters: list
            Query for filtering the search results
        page_size: int
            Size of each page of results
        page: int
            Page of the results
        asc: bool
            Whether to sort results by ascending or descending order
        flatten	: bool
            Whether to flatten
        log_ids: list
            The log dataset IDs to aggregate with - one or more of logs, logs-write, logs-search, logs-task or js-logs
        
        """
        return self.make_http_request(
            endpoint=f"/datasets/{dataset_id}/monitor/usage",
            method="POST",
            parameters={
                "filters": filters,
                "page_size": page_size,
                "page": page,
                "asc": asc,
                "filters": filters,
                "flatten": flatten,
                "log_ids": log_ids,
            },
            output_format=output_format,
            verbose=verbose,
        )
